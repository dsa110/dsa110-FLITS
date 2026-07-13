# In baseband_analysis/core/bbdata.py

"""Analysis data format for CHIME/FRB data."""

import io
import logging
import time
from glob import glob

import h5py
import numpy as np
import tenacity
from caput import memh5, tod

import baseband_analysis
from baseband_analysis.core import checks
from baseband_analysis.core._accel import unpack_baseband, unpack_baseband_transpose

log = logging.getLogger(__name__)


@tenacity.retry(
    wait=tenacity.wait_random(60, 120),
    stop=tenacity.stop_after_delay(300),
    reraise=True,
    retry=tenacity.retry_if_exception_type(IOError),
    before_sleep=tenacity.before_sleep_log(log, logging.WARNING),
)
def _get_h5py_File(fname, check_nans=False):
    f, opened = memh5.get_h5py_File(fname, mode="r")
    _ = list(f.attrs.items())
    _ = f.attrs["freq_id"]
    if check_nans:
        assert np.min(f["baseband"][:]) > 0, "Invalid values found in the channel"
    return f, opened


class BBData(memh5.BasicCont):
    """CHIME/FRB data in analysis format.

    Inherits from :class:`caput.memh5.BasicCont`.

    This is intended to be the main data class for the post
    acquisition/real-time analysis parts of the pipeline. This class is laid
    out very similarly to how the data is stored in analysis format hdf5 files
    and the data in this class can be optionally stored in such an hdf5 file
    instead of in memory.

    Parameters
    ----------
    h5_data : h5py.Group, memh5.MemGroup or hdf5 filename, optional
        Underlying h5py like data container where data will be stored. If not
        provided a new :class:`caput.memh5.MemGroup` instance will be created.

    Attributes
    ----------
    attrs
    baseband
    freq
        Central frequencies, in MHz.
    input
        Correlator input dataset.
    nfreq : int
        Size of the frequency axis.
    ninput : int
        Size of the correlator-input axis.
    ntime : int
        Size of the time axis.
    time
        Time offsets, in seconds.

    Methods
    -------
    from_file
    from_acq_h5
    create_dataset
    close
    flush
    save
    sort_inputs
    to_disk
    to_memory

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_file(
        cls,
        filename,
        check_conjugate=True,
        overwrite_if_not_conjugate=False,
        *args,
        **kwargs,
    ):
        """
        Load a data object from analysis hdf5 file, store in memory or on disk.

        If `ondisk` is True, do not load into memory but store data in h5py objects
        that remain associated with the file on disk. Check that the conjugation of
        beamformed data follows the expected convention.

        Parameters
        ----------
        filename : string or :class:`h5py.Group` object
            File with the hdf5 data. File must be compatible with memh5 objects.
        overwrite_if_not_conjugate: boolean, optional
            Whether beamfomed data should be overwritten if the conjugate
            convenction is the opposite than that used in the code
        ondisk : boolean, optional
            Whether the data should be stored in-place (in `filename`) or
            should be copied into memory.
        distributed : boolean, optional
            Allow the container to hold distributed datasets.
        comm : MPI.Comm, optional
            MPI Communicator to distributed over. If not set, use :obj:`MPI.COMM_WORLD`.
        detect_subclass: boolean, optional
            If *data_group* is specified, whether to inspect for a
            '__memh5_subclass' attribute which specifies a subclass to return.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to unicode. If not
            specified, look up the name as a class attribute to find a default,
            and otherwise use `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to unicode. If not specified,
            look up the name as a class attribute to find a default, and
            otherwise use `False`.
        <axis_name>_sel : list or slice
            Axis selections can be given to only read a subset of the containers.
            A slice can be given, or a list of specific array indices for that axis.
        **kwargs : any other arguments
            Any additional keyword arguments are passed to :class:`h5py.File`'s
            constructor if `filename` is a filename and silently ignored otherwise.
        """
        data = super().from_file(filename, *args, **kwargs)

        # Check whether data conjugation follows the expected convention
        if "tiedbeam_baseband" in data.keys():
            if check_conjugate and (
                ("conjugate_beamform" not in data["tiedbeam_baseband"].attrs)
                or (not data["tiedbeam_baseband"].attrs["conjugate_beamform"])
            ):
                log.warning(
                    "The beamformed baseband data is not properly formatted, "
                    "it will be conjugated for compatibility"
                )
                data["tiedbeam_baseband"][:] = np.conj(data["tiedbeam_baseband"][:])
                data["tiedbeam_baseband"].attrs["conjugate_beamform"] = int(1)
                if overwrite_if_not_conjugate:
                    log.warning(
                        "The old data will be overwritten with the "
                        "conjugated data in 10 seconds"
                    )
                    time.sleep(10)
                    data_temp = super().from_file(
                        filename, ondisk=True, *args, **kwargs
                    )
                    data_temp["tiedbeam_baseband"][:] = data["tiedbeam_baseband"][:]
                    data_temp["tiedbeam_baseband"].attrs["conjugate_beamform"] = int(1)
                    data_temp.close()

        # Check whether `phase_center` exists in existing singlebeam files
        # Rename `phase_center` to `centroid` if so
        if ("phase_center" in data.keys()) and ("centroid" not in data.keys()):
            log.warning(
                "The dataset named `phase_center` will be renamed to `centroid`"
            )
            entry = data["phase_center"]
            data.create_dataset(
                "centroid",
                shape=entry.shape,
                dtype=entry.dtype,
                data=np.ndarray.copy(entry[:], order="A"),
                chunks=entry.chunks,
                compression=entry.compression,
                compression_opts=entry.compression_opts,
            )
            memh5.copyattrs(entry.attrs, data["centroid"].attrs, convert_strings=True)
            del data["phase_center"]

        return data

    @classmethod
    def from_acq_h5(cls, acq_files, is_bytesIO=False, **kwargs):
        """Convert acquisition format hdf5 data to analysis data object.

        Reads hdf5 data produced by the acquisition system and converts it to
        analysis format in memory.

        Parameters
        ----------
        acq_files : filename, `h5py.File` or list there-of or filename pattern
            Files to convert from acquisition format to analysis format.
            Filename patterns with wild cards (e.g. "foo*.h5") are supported.
        is_bytesIO : boolean, optional
            Are the files stored in a bytesIO stream?
            Default False. Meaning that the files are stored locally on a local
            file system.
        transpose : boolean, optional
            Default True to unpack baseband using transpose when converting
            acquisition file to a standard analysis format file.
        out_group : `h5py.Group`, hdf5 filename or `memh5.Group`, optional
            Underlying hdf5 like container that will store the data for the
            BaseData instance.

        Returns
        -------
        data : BBData
            Loaded data object.

        Examples
        --------
        Suppose we have two acquisition format files:

        >>> path = '/data/frb-archiver/baseband_test/B0329_180924/'
        >>> filenames = [path+'event_1537874101_0652.h5',
                path+'event_1537874101_0655.h5']

        They can be converted into one big analysis format data object:

        >>> data = BBData.from_acq_h5(filenames)
        >>> print(data.keys())
        [u'baseband', 'time0', 'first_packet_recv_time']

        The underlying hdf5-like container that holds the *analysis format*
        data can also be specified.

        >>> group = memh5.MemGroup()
        >>> data = BBData.from_acq_h5(filenames, out_group=group)
        >>> print(group.keys())
        [u'baseband', 'time0', 'first_packet_recv_time', u'index_map', u'history']
        >>> group['baseband'] is a['baseband']
        True

        """
        if is_bytesIO is False:
            all_acq_files = tod.ensure_file_list(acq_files)
        else:
            if isinstance(acq_files, io.bytesIO):
                all_acq_files = [acq_files]
            elif memh5.is_group(files):
                all_acq_files = [acq_files]
            else:
                all_acq_files = acq_files

        if not len(all_acq_files):
            raise ValueError("Acquisition file list is empty.")

        try:
            # Open the files while keeping track of this so that we can close
            # them later.
            to_close = []
            acq_files = []
            for ii in range(len(all_acq_files)):
                try:
                    f, opened = _get_h5py_File(all_acq_files[ii])
                    acq_files.append(f)
                    to_close.append(opened)
                except Exception as e:
                    log.error(e)
                    log.error("Skipping file")
                    continue
            if not acq_files:
                raise OSError("All data corrupt.")

            data = cls._interpret_and_read(acq_files, **kwargs)

        finally:
            # Close any files opened in this function.
            for ii in range(len(acq_files)):
                if len(to_close) > ii and to_close[ii]:
                    acq_files[ii].close()

        return data

    @classmethod
    def _interpret_and_read(cls, acq_files, out_group=None, transpose=True):
        return concatenate(
            [cls(f) for f in acq_files],
            out_group=out_group,
            filter_func=lambda d: convert_acq(d, transpose=transpose),
            freq_sorter=lambda d: [d.attrs["freq_id"]],
        )

    _dataset_spec = {
        "baseband": {
            "axes": ["freq", "input", "time"],
            "dtype": np.complex64,
            "initialise": False,  # no idea what this means, -CL
            "distributed": True,
            "distributed_axis": "freq",
        },
        "time0": {
            "axes": ["freq"],
            "dtype": np.dtype(
                [("fpga_count", "<u8"), ("ctime", "<f8"), ("ctime_offset", "<f8")]
            ),  # custom dtype
            "initialise": False,  # no idea what this means, -CL
            "distributed": False,
        },
        "tiedbeam_locations": {
            "axes": ["beam"],
            "dtype": np.dtype(
                [
                    ("ra", "<f8"),
                    ("dec", "<f8"),
                    ("x_400MHz", "<f8"),
                    ("y_400MHz", "<f8"),
                    ("pol", "S1"),
                    ("source_name", "<S50"),
                ]
            ),  # custom dtype
            "initialise": False,  # no idea what this means, -CL
            "distributed": False,
        },
        "tiedbeam_baseband": {
            "axes": ["freq", "beam", "time"],
            "dtype": np.complex64,  # 128?
            "initialise": False,  # no idea what this means, -CL
            "distributed": False,
        },
        "tiedbeam_power": {
            "axes": ["freq", "beam", "time"],
            "dtype": np.float64,
            "initialise": False,  # no idea what this means, -CL
            "distributed": False,
        },
        "centroid": {
            "axes": ["freq", "xyz"],
            "dtype": np.float64,  #
            "initialise": False,  # no idea what this means, -CL
            "distributed": False,
        },
        "phase_center": {
            "axes": ["freq", "xyz"],
            "dtype": np.float64,  #
            "initialise": False,  # no idea what this means, -CL
            "distributed": False,
        },
    }

    @classmethod
    def _make_selections(cls, sel_args):
        """
        Match down-selection arguments to axes of datasets.

        Parses sel_* argument and returns dict mapping dataset names to selections.

        Parameters
        ----------
        sel_args : dict
            Should contain valid numpy indexes as values and axis names (str) as keys.
        Returns
        -------
        dict
            Mapping of dataset names to numpy indexes for downselection of the data.
            Also includes another dict under the key "index_map" that includes
            the selections for those.
        """
        # Check if all those axes exist
        for axis in sel_args.keys():
            if axis not in [
                "freq",
                "input",
                "time",
                "beam",
                "dimension",
                "xyz",
            ]:  # all the axes expected in BBData
                raise RuntimeError(f"No '{axis}' axis found to select from.")

        # Build selections dict
        selections = {}
        for name, dataset in cls._dataset_spec.items():
            ds_axes = dataset["axes"]
            sel = []
            ds_relevant = False
            for axis in ds_axes:
                if axis in sel_args:
                    sel.append(sel_args[axis])
                    ds_relevant = True
                else:
                    sel.append(slice(None))
            if ds_relevant:
                selections["/" + name] = tuple(sel)

        # add index maps selections
        for axis, sel in sel_args.items():
            selections["/index_map/" + axis] = sel

        return selections

    def copy(self, shared=None):
        """
        Makes a copy of a BBData, a non-nested MemGroup, which deepcopies its datasets.
        Could be generalized to MemGroups whose datasets are not all numpy arrays.
        """
        new = self.__class__(distributed=self.distributed, comm=self.comm)
        _copy(self, new, shared=shared)

        # Also copy the index map.
        for k in self.index_map.keys():
            new.create_index_map(axis_name=k, index_map=self.index_map[k])
        return new

    @property
    def ntime(self):
        """Length of time axis."""
        try:
            ntime = len(self.index_map["time"])
        except KeyError:
            a = list(self.baseband.attrs["axis"])
            time_axis = a.index("time")
            ntime = self.baseband.shape[time_axis]
        return ntime

    @property
    def nfreq(self):
        """Length of frequency axis."""
        try:
            nfreq = len(self.index_map["freq"])
        except KeyError:
            nfreq = 1
        return nfreq

    @property
    def ninput(self):
        """Correlator input number."""
        return len(self.input)

    @property
    def baseband(self):
        """Baseband data."""
        return self["baseband"]

    @property
    def freq(self):
        """Central frequencies, in MHz."""
        return self.index_map["freq"]["centre"]

    @property
    def time(self):
        """Time offsets, in seconds."""
        return self.index_map["time"]["offset_s"]

    @property
    def input(self):
        """Correlator input dataset."""
        return self.index_map["input"]

    def sort_inputs(self):
        """Reorder the correlator inputs by channel ID in-place."""
        sorter = np.argsort(self.index_map["input"]["chan_id"])
        self.baseband[:] = self.baseband[:, sorter, :]
        self.input[:] = self.input[sorter]

    def check(self):
        """Check this instance's contents against our expectations.

        This method tries to automatically determine whether this instance
        contains beamformed or acquisition ("raw"/"channelized") data and then
        compares its structure to expectations defined in the `core.checks` module.

        If the contents are deemed valid, nothing is returned. Otherwise, an
        exception will be raised.

        Raises
        ------
        ValueError
            If the instance is missing expected (meta)data, if those data
            have unexpected types, or if this routine is unable to determine
            whether the instance contains beamformed or acquisiton data.
        """
        if "tiedbeam_baseband" in self:
            checks.check_bbdata_singlebeam(self)
        elif "baseband" in self:
            checks.check_bbdata_acquisition(self)
        else:
            raise ValueError(
                "expecting either 'baseband' or 'tiedbeam_baseband' keys; "
                "unable to determine appropriate data format to check against"
            )


def concatenate(data_list, out_group=None, filter_func=None, freq_sorter=None):
    """Concatenate BBData objects over the frequency axis."""
    if len(data_list) == 0:
        raise ValueError("data_list cannot be empty")

    if filter_func is None:

        def filter_func(d):
            return d

    if freq_sorter is False:
        tosort = False
        freq_sorter = None
    else:
        tosort = True
    if freq_sorter is None:

        def freq_sorter(d):
            return list(d.index_map["freq"]["id"])

    # Presort, for efficiency.
    if tosort:
        data_list.sort(key=lambda d: freq_sorter(d)[0])

    ntime_max = 0
    nfreq = 0
    freq_id_list = []
    dset_shape = {}
    for d in data_list:
        freq_id_list += freq_sorter(d)
        nfreq += d.nfreq
        if d.ntime > ntime_max:
            ntime_max = d.ntime

        dataset_names = [v[0] for v in list(d.items()) if not memh5.is_group(v[1])]
        for name in dataset_names:

            dataset = d[name]
            axis = dataset.attrs["axis"]
            shape = list(dataset.shape)
            for ii in range(len(axis)):
                if axis[ii] == "time":
                    try:
                        if shape[ii] > dset_shape[name]["time"]:
                            dset_shape[name]["time"] = shape[ii]
                    except KeyError:
                        try:
                            dset_shape[name]["time"] = shape[ii]
                        except KeyError:
                            dset_shape[name] = {"time": shape[ii]}
                if axis[ii] == "freq":
                    try:
                        dset_shape[name]["freq"] += shape[ii]
                    except KeyError:
                        try:
                            dset_shape[name]["freq"] = shape[ii]
                        except KeyError:
                            dset_shape[name] = {"freq": shape[ii]}
                if axis[ii] == "beam":
                    try:
                        dset_shape[name]["beam"] = shape[ii]
                    except KeyError:
                        try:
                            dset_shape[name]["beam"] = shape[ii]
                        except KeyError:
                            dset_shape[name] = {"beam": shape[ii]}

    freq_sort_inds = np.argsort(np.argsort(freq_id_list))
    if np.all(np.diff(freq_sort_inds) == 1):
        tosort = False

    first = True
    freq_ind = 0
    for data in data_list:
        data = filter_func(data)
        if first:
            out = data.__class__(out_group)
            memh5.copyattrs(data.attrs, out.attrs)
            out.attrs["baseband-analysis_git_sha"] = baseband_analysis.__git_sha__
            for name, m in list(data.index_map.items()):
                if name not in ("freq", "time"):
                    out.create_index_map(name, m)
            freq_map = np.empty(nfreq, dtype=data.index_map["freq"].dtype)
            out.create_index_map("freq", freq_map)

            dataset_names = [
                v[0] for v in list(data.items()) if not memh5.is_group(v[1])
            ]
            for name in dataset_names:
                dataset = data[name]
                axis = dataset.attrs["axis"]
                shape = list(dataset.shape)
                for ii in range(len(axis)):
                    if axis[ii] == "time":
                        shape[ii] = dset_shape[name]["time"]
                    if axis[ii] == "freq":
                        try:
                            shape[ii] = dset_shape[name]["freq"]
                        except KeyError:
                            shape[ii] = nfreq
                    if axis[ii] == "beam":
                        shape[ii] = dset_shape[name]["beam"]
                out_dset = out.create_dataset(name, shape=shape, dtype=dataset.dtype)
                memh5.copyattrs(dataset.attrs, out_dset.attrs)
            first = False
        if data.ntime == ntime_max:
            out.create_index_map("time", data.index_map["time"])

        freq_sl = np.s_[freq_ind : freq_ind + data.nfreq]
        if tosort:
            freq_sl = freq_sort_inds[freq_sl]
        out.index_map["freq"][freq_sl] = data.index_map["freq"]
        for name in dataset_names:
            dataset = data[name]
            axis = dataset.attrs["axis"]
            sl = [np.s_[:]] * len(axis)
            taxis = -1
            shape = list(dataset.shape)
            for ii in range(len(axis)):
                if axis[ii] == "time":
                    taxis = ii
                    sl[ii] = np.s_[: shape[ii]]
                if axis[ii] == "freq":
                    sl[ii] = freq_sl
            out[name][tuple(sl)] = dataset
            if taxis != -1:
                sl[taxis] = np.s_[dataset.shape[taxis] :]
                out[name][tuple(sl)] = dataset.attrs["fill_value"]
        freq_ind += data.nfreq
    return out


def convert_acq(data, transpose=True, **kwargs):
    """Convert an acquisition file to a standard analysis format file."""
    # Read everything from disk.
    data = data.to_memory()

    # Add a frequency axis to all datasets.
    dataset_names = [v[0] for v in list(data.items()) if not memh5.is_group(v[1])]
    for name in dataset_names:
        d = data[name][:]
        a = dict(data[name].attrs)
        if name == "baseband":
            if transpose:
                a["axis"] = np.roll(a["axis"], -1)
                if data.ninput % 64 == 0:
                    d = unpack_baseband_transpose(d)
                else:
                    d = unpack_baseband(d).T
            else:
                d = unpack_baseband(d)
            a["fill_value"] = np.complex64(np.nan)
        d.shape = (1,) + d.shape
        del data[name]
        dset = data.create_dataset(name, data=d)
        memh5.copyattrs(a, dset.attrs)
        dset.attrs["axis"] = np.array(["freq"] + [str(at) for at in dset.attrs["axis"]])

    # Convert some attributes to datasets.
    freq_map = np.array(
        [(data.attrs["freq"], data.attrs["freq_id"])],
        dtype=[("centre", np.float64), ("id", np.uint32)],
    )
    data.create_index_map("freq", freq_map)
    del data.attrs["freq"]
    del data.attrs["freq_id"]

    time_axis = list(data["baseband"].attrs["axis"]).index("time")
    ntime = data["baseband"].shape[time_axis]
    time_map = np.empty(
        ntime, dtype=[("offset_fpga", np.uint64), ("offset_s", np.float64)]
    )
    time_map["offset_fpga"] = np.arange(ntime)
    try:
        time_map["offset_s"] = np.arange(ntime) * data.attrs["delta_time"]
    except KeyError:
        data.attrs["delta_time"] = data.attrs["fpga_sime"]
        time_map["offset_s"] = np.arange(ntime) * data.attrs["delta_time"]
    data.create_index_map("time", time_map)

    time0 = np.array(
        [
            (
                data.attrs["time0_fpga_count"],
                data.attrs["time0_ctime"],
                data.attrs["time0_ctime_offset"],
            )
        ],
        dtype=[
            ("fpga_count", np.uint64),
            ("ctime", np.float64),
            ("ctime_offset", np.float64),
        ],
    )
    dset = data.create_dataset("time0", data=time0)
    dset.attrs["axis"] = np.array(["freq"])
    del data.attrs["time0_fpga_count"]
    del data.attrs["time0_ctime"]
    del data.attrs["time0_ctime_offset"]

    first_packet_recv_time = np.array([data.attrs["first_packet_recv_time"]])
    dset = data.create_dataset("first_packet_recv_time", data=first_packet_recv_time)
    dset.attrs["axis"] = np.array(["freq"])
    del data.attrs["first_packet_recv_time"]

    return data


def add_weight_map(data, **maps):
    print("Deprecated function. Weights should be re-calculated each time.")
    return


def update_h5_parameters(file_to_update, file_to_read):
    print("Deprecated, recalculate parameters each time.")
    return


def merge_freq(files_pattern, output_filename):
    """
    Merge files over frequency axis.

    Parameters
    ----------
    files_pattern: str
        glob pattern to find input files.
    output_fileanme: str
        Output file path.

    """
    files_to_merge = glob(files_pattern)
    d_list = [BBData.from_file(n) for n in files_to_merge]
    merged = concatenate(d_list)
    if merged.freq.min() == 0:
        raise RuntimeError(
            "The files to be merged contain invalid channels containing frequencies labelled as 0 MHz."
        )
    merged.to_disk(output_filename)
    log.info(f"File {output_filename} written to disk")
    return


def reorder_inputs(data):
    """Temporary wrapper around BBData.sort_inputs method for compatibility.

    Note that this method reorders inputs in-place (the return is unnecessary)!
    """
    data.sort_inputs()
    return data


def _copy(
    g1,
    g2,
    shared=None,
    selections=None,
    convert_dataset_strings=False,
    convert_attribute_strings=True,
):
    memh5.copyattrs(g1.attrs, g2.attrs, convert_strings=True)
    for key in g1.keys():
        entry = g1[key]
        if memh5.is_group(entry):
            g_key = g2.create_group(key)
            _copy(
                entry,
                g_key,
                selections,
                convert_dataset_strings=convert_dataset_strings,
                convert_attribute_strings=convert_attribute_strings,
            )
        else:
            try:
                selection = selections.get(
                    entry.name, selections.get(entry.name[1:], slice(None))
                )
            except AttributeError:
                selection = slice(None)

            if convert_dataset_strings:
                # Convert unicode strings back into ascii byte strings. This will break
                # if there are characters outside of the ascii range
                if isinstance(g2, h5py.Group):
                    data = memh5.ensure_bytestring(entry[selection])

                # Convert strings in an HDF5 dataset into unicode
                else:
                    data = memh5.ensure_unicode(entry[selection])

            elif isinstance(g2, h5py.Group):
                data = memh5.check_unicode(entry)
                data = data[selection]
            else:
                data = entry[selection]

            if not (shared and key in shared):
                try:
                    g2.create_dataset(
                        key,
                        shape=data.shape,
                        dtype=data.dtype,
                        data=np.ndarray.copy(entry[:], order="A"),
                        chunks=entry.chunks,
                        compression=entry.compression,
                        compression_opts=entry.compression_opts,
                    )
                except AttributeError:
                    UserWarning(f"Copy is not deep for attribute {key}")
                    g2.create_dataset(
                        key,
                        shape=data.shape,
                        dtype=data.dtype,
                        data=data,
                        chunks=entry.chunks,
                        compression=entry.compression,
                        compression_opts=entry.compression_opts,
                    )
            else:
                g2.create_dataset(
                    key,
                    shape=data.shape,
                    dtype=data.dtype,
                    data=data,
                    chunks=entry.chunks,
                    compression=entry.compression,
                    compression_opts=entry.compression_opts,
                )

            memh5.copyattrs(
                entry.attrs, g2[key].attrs, convert_strings=convert_attribute_strings
            )



# In baseband_analysis/core/dedispersion.py

"""Methods for dedispersion of baseband data."""

import numpy as np
from chime_frb_constants import FPGA_DELTA_FREQ_MHZ, K_DM
from scipy.fft import fft, fftfreq, ifft


def coherent_dedisp(
    data,
    DM=None,
    matrix_in=None,
    ctime=None,
    frequencies=None,
    time_shift=False,
    f_ref=400,  # MHz
    t_ref=None,
    write=False,
):
    """
    Coherently de-disperse data.

    Parameters
    ----------
    data: `BBData`
        Data to de-disperse.
    DM: `float`
        Dispersion-measure to apply (pc/cc).
        Must pass in a DM. TODO: What if the data has a DM attribute already?
    matrix_in: `ndarray`, None
        Baseband data to dedisperse.
        Defaults to data['tiedbeam_baseband'].
    ctime: `ndarray`, None
        Starting time of each channel [s].
    frequencies: `ndarray`, None
        Central values [MHz] of each channel.
    time_shift: `bool`, False
        Shift the signal in each channel to align dump indices across the band.
        TODO: What if this gets written?
    t_ref: `float`, None
        Reference time [s] to shift the signal.
        Defaults to the lowest channel.
    f_ref: `float`, default 400 MHz
        Reference frequency (MHz) to shift the signal.
    write: `bool`, False
        Write dedispersed array to data['tiedbeam_baseband'] and
        DM to data['tiedbeam_baseband'].attrs['DM']

    Returns
    -------
    dedispersed_array: ndarray of complex128
        De-dispersed baseband data.

    TODO: The function could be speed up by cythonize the loop
    """
    if matrix_in is None:
        matrix_in = data["tiedbeam_baseband"][:]
    matrix_in = matrix_in.copy()
    if "DM" in data["tiedbeam_baseband"].attrs.keys():
        DM0 = data["tiedbeam_baseband"].attrs[
            "DM"
        ]  # data has already been de-smeared before.
    else:
        DM0 = 0  # data is not de-smeared at all

    DM_delta = DM - DM0
    print(DM_delta, DM, DM0)
    if frequencies is None:
        f0 = data.index_map["freq"]["centre"]
    else:
        f0 = frequencies
    if t_ref is None:
        t_ref = data["time0"]["fpga_count"][-1].astype(float)
    if ctime is None:
        ctime = (data["time0"]["fpga_count"].astype(float) - t_ref) * data.attrs[
            "delta_time"
        ]

    dedispersed_array = np.zeros_like(matrix_in) + np.nan
    for i, chan in enumerate(matrix_in):
        try:
            idx = np.where(np.isnan(chan.sum(axis=0)))[0][0]
        except IndexError:
            idx = None
        if idx == 0:
            # If the first value is nan, the whole channel is corrupt
            dedispersed_array[i, :] = np.nan
        else:
            chan_clean = chan[..., :idx]
            if DM_delta == 0:
                dedispersed_array[i, :, :idx] = chan_clean
            else:
                f = fftfreq(chan_clean.shape[-1], d=data.attrs["delta_time"] * 1e6)
                shift = (
                    +2j
                    * np.pi
                    * 1e6
                    * K_DM
                    * DM_delta
                    * f**2
                    / (f + f0[i])
                    / f0[i] ** 2
                )  # Intrachannel de-dispersion; make sure we use DM_delta
                if time_shift:
                    shift += (
                        -2j
                        * np.pi
                        * 1e6
                        * (f + f0[i])
                        * (
                            ctime[i]
                            - K_DM
                            * DM_delta
                            * (1 / f0[i] ** 2 - 1 / f_ref**2)  # noqa
                        )
                    )
                # Time shift to align the signal,
                # not recommended but implemented anyway...
                H = np.exp(shift)[np.newaxis]

                dedispersed_array[i, :, :idx] = ifft(fft(chan_clean) * H)
    if write:
        assert not time_shift, "Cannot save if time_shift == True!"
        data["tiedbeam_baseband"][:] = dedispersed_array
        data["tiedbeam_baseband"].attrs["DM"] = DM  # save total, not delta DM
    return dedispersed_array


def _coherent_dedisp(ctime, f0, matrix_in, DM, ntime):
    """Pythonify TODO: Tomas or Daniele or Calvin."""


def get_freq(data, factor=1):
    """Find frequencies for BBData instance."""
    f = data.index_map["freq"]["centre"]
    if factor % 2 == 0:
        shift = (np.arange(-factor // 2, factor // 2) + 0.5) / factor
    else:
        shift = (np.arange(-factor // 2, factor // 2) + 1.0) / factor
    shift *= FPGA_DELTA_FREQ_MHZ
    freq = np.ravel(f.repeat(factor).reshape([-1, factor]) + shift)
    f_id = sorted(data.index_map["freq"]["id"])
    freq_id = np.repeat(f_id, factor) * factor + np.tile(np.arange(factor), len(f_id))
    return freq_id, freq


def _get_freq(f, factor=1):
    """Pythonify TODO: anybody."""


def incoherent_dedisp(
    data,
    DM,
    factor=1,
    matrix_in=None,
    downsampled_time=1,
    fill_wfall=True,
    freq=None,
    f_ref=None,
    t_ref=None,
):
    """
    Apply incoherent dedispersion.

    Parameters
    ----------
    data: `BBData`
        Data to dedisperse.
    DM: float
        DM to apply for dedispersion.
    factor: int, default 1
       Time downsampling factor.
    matrix_in: `ndarray`
        Defaults to tiedbeam_baseband data.
    downsampled_time: int
    fill_wfall: bool
        If True, fill the waterfall with zeros for non-selected frequencies.
        Default True.
    freq: tuple (int or float, float)
        Frequency channel and frequency.

    Returns
    -------
    ndarray of complex128:
        Waterfall data.
    float:
        Frequency
    int:
        Frequency channel id.

    """
    if matrix_in is None:
        matrix_in = data["tiedbeam_baseband"][:]
    matrix_in = matrix_in.copy()
    wfall = np.zeros_like(matrix_in)

    if freq is None:
        f_id, f = get_freq(data, factor=factor)
    else:
        f_id, f = freq
    if f_ref is None:
        f_ref = min(f)
    f_id_ref = f_id[f_ref == f]

    if t_ref is None:
        t_ref = (
            data["time0"]["ctime"][f_id_ref == f_id]
            + data["time0"]["ctime_offset"][f_id_ref == f_id]
        )

    dt = data.attrs["delta_time"] * downsampled_time

    for i in range(matrix_in.shape[0]):
        j = i // factor
        t_shift = data["time0"]["ctime"][j] + data["time0"]["ctime_offset"][j] - t_ref
        start_t = t_shift + delay_across_the_band(DM, f_ref, f[i])
        bins_shift = np.round(start_t / dt / factor).astype(int)
        wfall[i] = np.roll(matrix_in[i], bins_shift, axis=-1)

    if fill_wfall:
        wfall_filled = np.zeros(
            (f_id.max() + 1,) + (wfall.shape[1], wfall.shape[2]), dtype=wfall.dtype
        )
        wfall_filled[f_id] = wfall
        return wfall_filled, f, f_id
    else:
        return wfall, f, f_id


def _incoherent_dedisp(ctime, f0, matrix_in, DM, factor=1):
    """Pythonify TODO: Tomas, Daniele, or Calvin."""


def delay_across_the_band(
    DM: float, freq_low: float = 400.390625, freq_high: float = 800
) -> float:
    """
    Return the delay in seconds caused by dispersion.

    It is assumed that Dispersion Measure (DM) in cm-3 pc, and the emitted
    frequency (freq_emitted) of the pulsar in MHz.

    Parameters
    ----------
    DM : float
        Dispersion measure.
    freq_low : float, optional
        Lowest observing frequency in MHz, by default 400
    freq_high : float, optional
        Highest observing frequency in MHz, by default 800

    Returns
    -------
    float
        Delay across the band.
    """
    return K_DM * DM * (1.0 / freq_low**2 - 1.0 / freq_high**2)


# In baseband_analysis/core/sampling

"""Methods for resampling baseband data."""

from decimal import Decimal

import astropy.units as u
import chime_frb_constants as constants
import numpy as np
from astropy.time import Time, TimeDelta
from caput.memh5 import copyattrs
from scipy.fft import fft, fftshift, irfft, next_fast_len, rfft, rfftfreq
from scipy.interpolate import interp1d

from baseband_analysis.core.dedispersion import delay_across_the_band
from baseband_analysis.core.fetch import get_measured_parameter


def clip(
    data_in,
    toa_400: float,
    ref_freq: float,
    dm: float,
    duration: float,
    pad: bool = False,
    inplace=True,
):
    """Clip beamformed baseband data around a pulse.

    The clipped data include the time of arrival ± `duration`/2. This method
    changes the BBData instance in-place, updating the "tiedbeam_baseband"
    dataset, "time" index map, and "time0" dataset accordingly.

    This method accomodates clip requests that are not strict subsets of the
    original BBData instance by passing `pad=True`. By default, `pad` is set to
    False to avoid accidentally allocating large arrays of NaNs (by, e.g.,
    passing the wrong TOA).

    TODO: add an `inplace` kwarg and support for clipping acq data

    Parameters
    ----------
    data : BBData
        BBData instance with beamformed baseband data to clip
    toa_400 : float
        UNIX time of arrival, referenced at 400MHz
    dm : float
        Dispersion measure of clipped data in pc cm**-3
    duration : float
        duration of the clipped baseband, in seconds
    pad : bool
        Pad the original data with the dataset's `fill_value` to achieve the
        requested output size. Otherwise, raise a ValueError if the requested
        clip includes samples outside the original range.
    inplace : bool
        Perform the clipping in-place on this BBData instance.
        Otherwise, return a new BBData instance with the clipped data."""
    if inplace:
        data = data_in
    else:
        data = data_in.copy()

    if "tiedbeam_baseband" not in data:
        raise NotImplementedError("this method only works with beamformed data")

    if dm is None:
        dm = data["tiedbeam_baseband"].attrs["DM"]
    freq = data.index_map["freq"]["centre"]
    delay = delay_across_the_band(
        dm,
        freq_low=ref_freq,
        freq_high=freq,
    )
    toa = toa_400 - delay  # TOA for each frequency

    baseband, start_inds, ctime, ctime_offset = _clip(
        data["tiedbeam_baseband"],
        data.index_map["freq"]["centre"],
        data["time0"]["ctime"],
        data["time0"]["ctime_offset"],
        toa,
        duration,
        dt=data.attrs["delta_time"],
        pad=pad,
        pad_value=data["tiedbeam_baseband"].attrs["fill_value"],
    )

    ntime = baseband.shape[2]

    # TODO: deal with first_packet_recv_time too?
    baseband_attrs = data["tiedbeam_baseband"].attrs
    data.create_dataset("tiedbeam_baseband", data=baseband)
    data["tiedbeam_baseband"].attrs.update(baseband_attrs)

    index_map_time = data.index_map["time"][:ntime]
    fpga_count = (data["time0"]["fpga_count"].astype(int) + start_inds).astype(
        np.uint64
    )
    data.create_index_map("time", index_map_time)
    data["time0"]["ctime"] = ctime
    data["time0"]["ctime_offset"] = ctime_offset
    data["time0"]["fpga_count"] = fpga_count

    if "tiedbeam_power" in data:
        del data["tiedbeam_power"]

    if "downsampled_time" in data.index_map:
        data.del_index_map("downsampled_time")
    return data


def _clip(
    baseband: np.ndarray,
    freq: np.ndarray,
    ctime: np.ndarray,
    ctime_offset: np.ndarray,
    toa: np.ndarray,
    duration: float,
    dt: float = 2.56e-6,
    pad: bool = False,
    pad_value: float = np.nan,
):
    """Clip baseband data array around a pulse.

    Parameters
    ----------
    baseband: array, shape (nfreq, ninput, ntime)
    freq: array, shape (nfreq)
        Channels' central frequencies, in MHz.
    ctime: array, shape (nfreq)
        UNIX times of first frame in each channel
    ctime_offset: array, shape (nfreq)
    toa: array, shape (nfreq)
        Time of arrival in each channel.
    duration: float
        Duration of the clipped baseband, in seconds.
    dt: float
        Sample spacing, in seconds.
    pad: bool
        Pad the original data with `pad_value` to achieve the requested output
        size. Otherwise, raise a ValueError if the requested clip includes
        samples outside the original range.
    pad_value: float
        Value used to pad the input data array.

    Returns
    -------
    clipped: array, shape (nfreq, ninput, ntime_new)
        Clipped baseband data array.
    start_inds: array, shape (nfreq)
        Indices of the first sample taken from each channel.
    ctime_new: array, shape (nfreq)
        ctime for each first sample in the new (clipped) array.
    ctime_offset_new: array, shape (nfreq)
        ctime_offset for each first sample in the new (clipped) array.

    Raises
    ------
    ValueError
        If the clip is not a subset of this BBData instance and `pad=False`,
        or if the requested clip is disjoint with this instance.
    """
    if duration > baseband.shape[2] * dt:
        raise ValueError("requesting duration longer than original dataset")

    # The ctime_offsets are much smaller than the sample spacing, so we'll
    # ignore them in this calculation (they'll get dropped by the float
    # precision anyway). This means we'll pick the "wrong" time samples
    # some fraction of the time (i.e. ones that don't precisely agree with
    # `toa_400` & `duration`) but that shouldn't matter, since we'll still
    # compute the appropriate times for those samples.
    start_times = toa - duration / 2
    start_inds = np.round((start_times - ctime) / dt).astype(int)

    nfreq, nbeam, ntime_orig = baseband.shape
    ntime = int(np.round(duration / dt))

    if np.all(start_inds >= ntime_orig) or np.all(start_inds + ntime < 0):
        raise ValueError("clipped region fully outside the bounds of this dataset")

    shape = (nfreq, nbeam, ntime)
    dtype_baseband = baseband.dtype  # probably complex64

    if pad:
        # lazy fill
        clipped = np.full(shape, pad_value, dtype_baseband)
    elif np.any(start_inds < 0) or np.any(start_inds + ntime > ntime_orig):
        raise ValueError("requested samples are out of bounds but `pad=False`")
    else:
        clipped = np.empty(shape, dtype_baseband)

    ctime_new = np.empty_like(ctime)
    ctime_offset_new = np.empty_like(ctime_offset)

    for ifreq, istart in enumerate(start_inds):
        # four cases we need to handle here:
        # first ind is OOB, last ind is OOB, both inds are OOB, neither are OOB
        jstart = max(min(istart, ntime_orig), 0)
        jend = max(min(istart + ntime, ntime_orig), 0)

        clipped[ifreq, :, jstart - istart : jend - istart] = baseband[
            ifreq, :, jstart:jend
        ]

        # compute "true" time with decimal math...
        ct_decimal = (
            Decimal(ctime[ifreq]) + Decimal(ctime_offset[ifreq]) + istart * Decimal(dt)
        )
        # ...but store `ctime` and `ctime_offset` as floats
        ctime_new[ifreq] = ct_decimal
        ctime_offset_new[ifreq] = ct_decimal - Decimal(ctime_new[ifreq])

    return clipped, start_inds, ctime_new, ctime_offset_new


def fpga_start_time(
    data, start_time_error=True, integer_error=False, astropy_time=True
):
    """Returns the FPGA start time, inferred from every channel of baseband data present, as a list of Decimal objects"""
    dump_time = Time(
        data["time0"]["ctime"],
        val2=data["time0"]["ctime_offset"],
        format="unix",
        precision=9,
    )
    fpga_start = dump_time - TimeDelta(2.56e-6 * u.s * data["time0"]["fpga_count"])
    fpga_start_unix = fpga_start.to_value("unix", subfmt="decimal")
    if (
        np.mod(np.median(fpga_start_unix), 1) >= 1e-9
    ):  # CHIME FPGAs start on an integer number of seconds , but not all F engines do
        if integer_error:
            ValueError("FPGAs do not start on an integer second")
        else:
            UserWarning("FPGAs do not start on an integer second")

    if (
        np.max(fpga_start_unix) - np.min(fpga_start_unix) > 1e-9
    ):  # should be within one FPGA cycle
        if start_time_error:
            ValueError("Frequency channel timestamps differ by more than 1 nanosecond!")
        else:
            UserWarning(
                "Frequency channel timestamps differ by more than 1 nanosecond!"
            )
    if astropy_time:
        return fpga_start
    else:
        return fpga_start_unix


def fill_waterfall(
    data, matrix_in: np.ndarray = None, f_id: np.ndarray = None, write: bool = False
):
    """Restore missing channels in baseband data, filling them with zeros.

    The BBData object is modified in place.
    Using dump times for successfully-captured frequencies, arrival times for the infilled channels are calculated by linear interpolation.
    data['tiedbeam_baseband'] contains 1024 freq channels after calling fill_wfall.
    data['first_packet_recv_time'] contains 1024 time stamps. We infill with values from data['time0']['ctime'] for missing channels.
    data['time0'] will have 1024 start times (in units of fpga_counts and ctime in unix seconds)
    data.index_map['freq'] will have 1024 frequencies.

    No other attributes are being modified.
    Each channel will be filled with zeros.
    If channels are added at the top or the bottom of the band (not in betwee two valid channels),
    Their dump times will be equal to the earliest/latest valid channel to preserve the behavior of incoherent_dedisp.

    """
    if f_id is None:
        f_id = data.index_map["freq"]["id"][:]
    if matrix_in is None:
        matrix_in = data["tiedbeam_baseband"][:]
        legal_to_write = True
    assert f_id.size == matrix_in.shape[0]

    matrix_in_filled = np.zeros(
        [1024] + list(matrix_in.shape[1:]), dtype=matrix_in.dtype
    )
    matrix_in_filled[f_id] = matrix_in

    if write and legal_to_write and len(f_id) < 1024:
        # Change data['time0']
        f_id = data.index_map["freq"]["id"].copy()
        t0_dtype = data["time0"].dtype
        new_t0 = np.zeros(1024, dtype=t0_dtype)
        new_first_packet_recv_time = np.zeros(
            1024, dtype=data["first_packet_recv_time"].dtype
        )
        # Construct linear interpolants:
        # First, fill in missing fpga_count entries with rounding and linear interpolation
        # Next, calculate fpga_start_time carefully with full (Decimal) precision.
        # Fill in fpga_start_time with nearest-neighbor interpolation.
        # Calculate ctime by casting fpga_start_time + fpga_count * 2.56e-6 to a float64
        # Calculate ctime_offset by looking at the difference between fpga_start_time and ctime.

        # interpolate and round to get the new fpga_count
        fpga_start_unix = fpga_start_time(
            data, integer_error=False, start_time_error=True, astropy_time=False
        )
        new_t0["fpga_count"] = np.round(
            np.interp(
                xp=data.index_map["freq"]["id"],
                fp=data["time0"]["fpga_count"],
                x=np.arange(1024),
                left=data["time0"]["fpga_count"][0],
                right=data["time0"]["fpga_count"][-1],
            )
        )  # FPGA count is always an integer!
        new_fpga_start_time_unix = interp1d(
            x=data.index_map["freq"]["id"],
            y=fpga_start_unix,
            kind="nearest",
            fill_value=(fpga_start_unix[0], fpga_start_unix[-1]),
            bounds_error=False,
        )  # Nearest neighbor interpolation for fpga_start_time. This should not really matter.
        # calculate new values for ctime and ctime_offset
        for freq_id, fpga_count, ftime in zip(
            np.arange(1024),
            new_t0["fpga_count"],
            new_fpga_start_time_unix(np.arange(1024)),
        ):
            ct_decimal = Decimal(ftime) + fpga_count * Decimal(2.56e-6)
            # ...but store `ctime` and `ctime_offset` as floats
            new_t0["ctime"][freq_id] = ct_decimal
            new_t0["ctime_offset"][freq_id] = ct_decimal - Decimal(
                new_t0["ctime"][freq_id]
            )
            new_first_packet_recv_time[freq_id] = ct_decimal
            # calculate time0 and time0 offset

        # ...but keep original values where possible
        for freq_id, fpga_count, ctime, ctime_offset, fpr_time in zip(
            data.index_map["freq"]["id"],
            data["time0"]["fpga_count"],
            data["time0"]["ctime"],
            data["time0"]["ctime_offset"],
            data["first_packet_recv_time"],
        ):
            new_t0["ctime"][freq_id] = ctime
            new_t0["ctime_offset"][freq_id] = ctime_offset
            new_t0["fpga_count"][freq_id] = fpga_count
            new_first_packet_recv_time[freq_id] = fpr_time
        # Make new frequency index map
        im_freq_dtype = data.index_map["freq"].dtype
        new_freq_map = np.zeros(1024, dtype=im_freq_dtype)
        new_freq_map["id"] = np.arange(1024)
        new_freq_map["centre"] = np.linspace(800, 400, num=1024, endpoint=False)

        # Update BBData
        for name, filled_data in zip(
            ["first_packet_recv_time", "time0", "tiedbeam_baseband"],
            [new_first_packet_recv_time, new_t0, matrix_in_filled],
        ):
            old_attrs = data[name].attrs
            filled_ds = data.create_dataset(name, data=filled_data)
            copyattrs(old_attrs, filled_ds.attrs)
        # Create new data.index_map['freq']
        data.create_index_map("freq", new_freq_map)

        # Run fpga_start_time to check that the start times are lined up
        fpga_start_unix = fpga_start_time(
            data, integer_error=False, start_time_error=True
        )

        print("Written to dataset and updated attributes.")
    return data, matrix_in_filled


def downsample_power(power, data, factor=None, peak_bins=5):
    """Downsample power array."""
    if factor is None:
        measured_parameter = get_measured_parameter(data.attrs["event_id"])
        width = measured_parameter["width"]
        factor = int(round(width / 1000.0 * constants.FPGA_FREQUENCY_HZ / peak_bins))
        if factor < 1:
            factor = 1
    return _downsample_power(power, factor)


def _downsample_power(power, factor):
    """Downsample over the last (time) axis."""
    power = power[..., : power.shape[-1] // factor * factor]
    power = np.sum(
        power.reshape(list(power.shape[:-1]) + [power.shape[-1] // factor, factor]),
        axis=-1,
    ) / np.sqrt(factor)
    return power, factor


def downsample_power_gaussian(power, data, factor=None, peak_bins=5, upsample=1):
    """Convolves flux profile with a gaussian of width :factor:, in units of frames.
    If :factor: is not provided, it is guessed from the L1 pipeline."""
    if factor is None:
        measured_parameter = get_measured_parameter(data.attrs["event_id"])
        width = measured_parameter["width"]
        factor = int(round(width / 1000.0 * FPGA_FREQUENCY_HZ / peak_bins))

    return _downsample_power_gaussian(power, factor, upsample)


def _downsample_power_gaussian(power, factor, upsample):
    # performs FFT convolution with a kernel
    nfft = next_fast_len(power.shape[-1])
    mask = ~np.isfinite(power[..., 0 : power.shape[-1]])
    k = rfftfreq(nfft)
    power_out = (
        upsample
        * irfft(
            rfft(power, axis=-1, n=nfft) * np.exp(-np.pi**2 * k**2 * factor**2),
            n=nfft * upsample,
            axis=-1,
        )[..., 0 : power.shape[-1] * upsample]
    )
    return power_out, factor


def upchannel(data, fftsize=32, downfreq=2):
    """Upchannel dynamic spectra."""
    return _upchannel(
        data["tiedbeam_baseband"][:],
        freq_id=data.index_map["freq"]["id"][:],
        fftsize=32,
        downfreq=2,
    )


def _upchannel(wfall, freq_id, fftsize=32, downfreq=2):
    """Upchannelize a dynamic spectrum.

    Performs the CHIME upchannelization on a dynamic spectrum,
    average every 3 time samples (hard-coded) and every `downfreq`
    frequency channels after upchannelization.

    Parameters
    ----------
    wfall : np.ndarray
        Dynamic spectrum to process.
    freq_id : np.1darray
        frequency channel ids
    fftsize : int
        FFT step-size.
    downfreq : int
        Downsampling factor in frequency.

    Returns
    -------
    upchan : np.ndarray[:, nfreq]
        Array of upchannelization frequencies, ordered high to low
        (order will change later!).
    """
    # swap axes ordering to (pol,time,chan)
    wfall = np.swapaxes(wfall, 0, 1)
    wfall = np.swapaxes(wfall, 1, 2)

    # set downtime to 1 => no averaging over complex numbers!!!
    downtime = 1

    npol, nsamp, nchan = wfall.shape

    # upchannelization factor (16 by default)
    upchan = fftsize // downfreq

    # number of blocks
    nblock = nsamp // (fftsize * downtime)

    # initialise array for spectrum
    spec = np.zeros((npol, nblock, nchan * upchan), dtype=np.complex64)

    # iterate over blocks and perform the upchannelization
    count = 0
    chan_id_upchan = np.zeros((nchan * upchan), dtype=int)
    f_upchan_bandtot = np.linspace(
        constants.FREQ_TOP_MHZ, constants.FREQ_BOTTOM_MHZ, upchan * 1024
    )
    for pol in range(npol):
        for bi in range(nblock):
            for chidx in range(nchan):
                # cut out the correct timestream section
                ts = wfall[pol, bi * fftsize : bi * fftsize + fftsize, chidx].copy()
                # perform a FFT
                ft = fftshift(fft(ts))
                # downsample in frequency
                ft = ft.reshape(upchan, downfreq).mean(axis=1).copy()

                spec[pol, bi, chidx * upchan : chidx * upchan + upchan] = ft

                chan_id_upchan[chidx * upchan : chidx * upchan + upchan] = np.arange(
                    upchan * freq_id[chidx], upchan * freq_id[chidx] + upchan, 1
                )

                count += 1

    return spec, f_upchan_bandtot[chan_id_upchan], chan_id_upchan


def _scrunch(wfall, tscrunch, fscrunch):
    """Return a rebinned array, useful for waterfall plots.

    Parameters
    ----------
    wfall: ndarray (2D)
        Array to be rebinned
    tscrunch: int
        scrunching factor along first (e.g. time) axis
    fscrunch: int
        scrunching factor along second (e.g. frequency) axis

    Returns
    -------
    rebinned_array: array

    """
    nbins = wfall.shape[-1]
    remainder = nbins % tscrunch
    wfall = wfall[..., : nbins - remainder]
    wfall = np.nanmean(
        wfall.reshape(wfall.shape[:-1] + (nbins // tscrunch, tscrunch)), axis=-1
    )
    nchan = wfall.shape[0]
    if nchan % fscrunch != 0:
        raise ValueError("Number of channel not an integer factor of fscrunch.")
    wfall = np.nanmean(
        wfall.reshape((nchan // fscrunch, fscrunch) + wfall.shape[1:]), axis=1
    )
    return wfall


def scrunch(wfall, tscrunch, fscrunch):
    """Return a rebinned array, useful for waterfall plots.

    Parameters
    ----------
    wfall: ndarray (2D)
        Array to be rebinned
    tscrunch: int
        scrunching factor along first (e.g. time) axis
    fscrunch: int
        scrunching factor along second (e.g. frequency) axis

    Returns
    -------
    rebinned_array: array

    """
    nbins = wfall.shape[-1]
    remainder = nbins % tscrunch
    wfall = wfall[..., : nbins - remainder]
    wfall = np.nanmean(
        wfall.reshape(wfall.shape[:-1] + (nbins // tscrunch, tscrunch)), axis=-1
    )
    nchan = wfall.shape[0]
    if nchan % fscrunch != 0:
        raise ValueError("Number of channel not an integer factor of fscrunch.")
    wfall = np.nanmean(
        wfall.reshape((nchan // fscrunch, fscrunch) + wfall.shape[1:]), axis=1
    )
    return wfall

# In baseband_analysis/core/signal.py 

"""General signal-processing methods for baseband data."""

import logging

import numpy as np
from scipy.optimize import curve_fit

from baseband_analysis.analysis.dm import DM_SNR
from baseband_analysis.core.dedispersion import (
    coherent_dedisp,
    delay_across_the_band,
    incoherent_dedisp,
)
from baseband_analysis.core.fetch import get_measured_parameter
from baseband_analysis.core.flagging import (
    get_RFI_channels,
    get_RFI_spacial,
    get_valid_time_range,
)
from baseband_analysis.core.sampling import downsample_power

log = logging.getLogger(__name__)


def squarewave_pattern_index(total_intensity, freq, dt=2.56e-6):
    """
    Calculate the starting index for each channel of
    the square-wave pattern introduced dumping the data.

    Parameters
    ----------
    total_intensity: `ndarray`
        Array of total intensity showing the square-wave pattern.
    freq: `ndarray`
        Frequency value [MHz] of single channels.
    dt: `float`, default = 2.56e-6
        Time resolution of the data [s].

    Returns
    -------
    idx: `ndarray`
        Index array.
    """
    if len(total_intensity.shape) > 2:
        total_intensity = total_intensity.sum(axis=1)
    if ~np.isnan(total_intensity).any():
        # Returns if there are no NaN values
        return (
            np.ones(total_intensity.shape[0], dtype=int) + total_intensity.shape[-1] - 1
        )
    y, x = np.where(np.isnan(total_intensity))
    x_0 = x.min()
    y_0 = y[x.argmin()]
    freq_0 = freq[y_0]

    idx = []
    t_p = np.inf
    t_bad = 0
    bad_chunk = False
    for i, n in enumerate(total_intensity):
        t = len(n[~np.isnan(n)])
        if t <= t_p:
            idx.append(i)
            t_p = t
    bins = np.array([len(n[~np.isnan(n)]) for n in total_intensity])

    def dm_delay(freq, dm):
        return delay_across_the_band(dm, freq_0, freq) / dt + x_0

    dm, _ = curve_fit(dm_delay, freq[idx], bins[idx], p0=[10])
    idx = np.floor(delay_across_the_band(dm, freq_0, freq) / dt + x_0).astype(int)
    return idx


def get_power(data, DM=None, diagnostic_plots=False, check_old_process=True):
    """
    Compute total power in baseband data.

    Parameters
    ----------
    data: `BBData`
        Baseband data.
    DM: float
        Dispersion measure.
    diagnostic_plots: bool
        Also make diagnostic plots (default False)
    check_old_process: bool
        Remove junk data in old processings
    Returns
    -------
    float:
        frequency
    int:
        frequency channel index.
    ndarray:
        Intensity data.

    TODO: Does not need pythonification because this is basically a wrapper
    around incoherent_dedisp, not _incoherent_dedisp, and this is not core functionality
    """
    if DM is None:
        DM = data["tiedbeam_power"].attrs["DM_coherent"]

    # Remove junk data in old processings
    power_in = data["tiedbeam_power"][:].copy()
    if not np.isnan(np.nanmean(power_in[-1], axis=0)).any():
        if not check_old_process:
            log.warning(
                "This may be an old event, it might help using check_old_process=True"
            )
        else:
            power_mean = np.nanmean(data["tiedbeam_power"][:], axis=1)
            freq = []
            bins = []
            for f, chan in zip(data.freq, power_mean):
                try:
                    bin_i = np.nonzero(np.diff(chan, axis=-1) > chan.max() / 1e3)[0][-1]
                    if bin_i < chan.size - 5:
                        bins.append(bin_i)
                        freq.append(f)
                except IndexError:
                    pass
            if len(bins) < 30:  # Ignore correction if acts on less than 30 channels
                log.warning("No junk data found, skipping old processing correction")
            else:
                bins = np.array(bins)
                freq = np.array(freq)
                dt = (
                    data.attrs["delta_time"]
                    * data["tiedbeam_power"].attrs["time_downsample_factor"]
                )

                def dm_delay(freq, dm):
                    return delay_across_the_band(dm, freq, freq.min()) / dt + bins[-1]

                dm, _ = curve_fit(dm_delay, freq, bins, p0=[10])
                idx = np.floor(
                    delay_across_the_band(dm, data.freq, freq.min()) / dt + bins[-1]
                ).astype(int)
                for i, n in enumerate(idx):
                    power_in[i, ..., n:] = np.nan

            # Remove beginning of dump
            diff = np.diff(power_mean, axis=-1)
            start = 0
            n_freq = 0
            for chan in diff:
                try:
                    start += np.nonzero(chan < 0)[0][0]
                    n_freq += 1
                except IndexError:
                    pass
            start_bin = int(start / n_freq) + 1
            power_in = power_in[..., start_bin:]

    power, freq, freq_id = incoherent_dedisp(
        data,
        DM,
        matrix_in=power_in,
        downsampled_time=data["tiedbeam_power"].attrs["time_downsample_factor"],
        fill_wfall=False,
    )

    total_intensity = np.sum(
        power.reshape([power.shape[0], power.shape[1] // 2, 2, power.shape[2]]), axis=2
    )
    try:
        total_intensity = total_intensity.squeeze(axis=1)
    except ValueError:
        pass
    return freq, freq_id, total_intensity


def frequency_average(power):
    """Average normalized power along the frequency axis."""
    return np.nanmean(power, axis=0) * np.sqrt(
        np.count_nonzero(~np.isnan(power), axis=0)
    )


def get_onsource_beam(power, check_time_range=False):
    """Find index of maximum power."""
    if check_time_range:
        time_range = get_valid_time_range(power)
        power = power[..., time_range[0] : time_range[1]]
    if len(power.shape) == 2:
        power = np.nanmax(power, axis=0)
    if len(power.shape) == 3:
        power = frequency_average(power)
    if len(power.shape) == 2:
        power = np.nanmax(power, axis=-1)
    return power.argmax()


def get_profile(power, min_channels=None):
    """Compute burst profile.
    Parameters
    ----------
    power : array_like
        Normalized power.
    min_channels : int, optional
        Minimum number of channels to compute a profile.
    """
    if len(power.shape) == 1:
        # only time axis is left
        return power
    elif len(power.shape) == 3:
        # freq, beam, time axes left
        power = power[:, get_onsource_beam(power)]
    profile = frequency_average(power)
    if min_channels is not None:
        # Only returns profile bins with enough statistics
        prof_channels = np.count_nonzero(~np.isnan(power), axis=0)
        profile = profile[prof_channels > 10]
    return profile


def get_main_peak_lim(
    prof, floor_level=0, diagnostic_plots=False, normalize_profile=False
):
    """
    Get indices surrounding the main peak.

    Parameters
    ----------
    prof: ndarray
        Profile of power, or the full waterfall.
    floor_level: float
        Minimum power level (default 0)
    diagnostic_plots: bool
        Also make diagnostic plots (default False)
    normalize_profile: bool
        Normalize profile to be in range +/- 1 (default False)

    Returns
    -------
    int, int:
        Lower and upper indices of peak region.
    """
    prof = prof.copy()
    # Select peak
    if len(prof.shape) > 1:
        prof = get_profile(prof)
    if normalize_profile:
        prof -= np.nanmedian(prof)
        prof /= np.nanmax(prof)
    try:
        peak_t0 = np.where(prof[: np.nanargmax(prof)] < floor_level)[0][-1]
    except (IndexError, ValueError):
        peak_t0 = 0
    try:
        peak_t1 = np.where(prof[np.nanargmax(prof) :] < floor_level)[0][
            0
        ] + np.nanargmax(prof)
    except (IndexError, ValueError):
        peak_t1 = prof.size

    if diagnostic_plots:
        from baseband_analysis.utilities import plotting

        if isinstance(diagnostic_plots, bool):
            save_dir = None
        else:
            save_dir = diagnostic_plots

        plotting.main_peak_lim_diagnostic(prof, peak_t0, peak_t1, save_dir=save_dir)

    return peak_t0, peak_t1


def get_floor(power, diagnostic_plots=False):
    """Get indices of power floor (noiselike to within 3 sigma)."""
    prof = get_profile(power)
    floor = prof.copy()
    floor -= np.nanmedian(floor)
    floor[np.isinf(floor)] = np.nan
    floor /= np.nanstd(floor)
    while True:
        peak_t0, peak_t1 = get_main_peak_lim(floor, floor_level=0)
        if (peak_t1 - peak_t0) == floor.size:
            break
        floor[peak_t0:peak_t1] = np.nan
        floor -= np.nanmedian(floor)
        floor[np.isinf(floor)] = np.nan
        floor /= np.nanstd(floor)
        idx = np.abs(floor) > 3  # Identify bins larger than 3 sigma
        floor[idx] = np.nan
        if len(idx[idx]) == 0:  # If no bins larger than 3 sigma
            break
        if len(floor[~np.isnan(floor)]) == 0:  # All bins larger than 3 sigma
            break
    idx = np.isnan(floor)

    if diagnostic_plots:
        from baseband_analysis.utilities import plotting

        if isinstance(diagnostic_plots, bool):
            save_dir = None
        else:
            save_dir = diagnostic_plots

        floor = power.copy()
        floor[..., idx] = np.nan

        plotting.floor_diagnostic(power, get_profile(floor), save_dir=save_dir)

    return idx


def get_spectrum(power, peak_lim=None):
    """Compute spectrum from baseband power."""
    if len(power.shape) == 3:
        power = power[:, get_onsource_beam(power)]
    if peak_lim is None:
        peak_t0, peak_t1 = get_main_peak_lim(power)
    else:
        peak_t0, peak_t1 = peak_lim
    return np.sum(power, axis=-1) / np.sqrt(peak_t1 - peak_t0)


def get_spectrum_lim(f_id, power, min_snr=2, diagnostic_plots=False):
    """
    Compute spectrum peak limits.

    Given an array with (freq, beam, time) axes, pick the brightest beam in a
    grid of formed beams, take its spectrum, and fit a Gaussian to the
    spectrum, and return the full width at 1/e power (as opposed to the FWHM)
    of the spectrum.
    """
    power = power.copy()
    idx_floor = get_floor(power, diagnostic_plots=diagnostic_plots)
    floor = power.copy()
    floor[..., idx_floor] = np.nan
    mean = np.nanmean(floor, axis=-1)[..., np.newaxis]
    floor -= mean
    floor[np.isinf(floor)] = np.nan
    std = np.nanstd(floor, axis=-1)[..., np.newaxis]
    power -= mean
    power /= std
    if len(power.shape) == 3:
        power = power[:, get_onsource_beam(power, check_time_range=True)]
    peak_t0, peak_t1 = get_main_peak_lim(power, floor_level=0.1, normalize_profile=True)
    power = power[..., peak_t0:peak_t1]

    spect = get_spectrum(power, [peak_t0, peak_t1])
    spect = np.nan_to_num(spect, posinf=np.sort(spect)[-2])
    idx_nan = ~np.isnan(spect)
    spect = spect[idx_nan]
    f = f_id[idx_nan].astype(float)

    # Fit a Gaussian to the spectrum
    def model_gaussian(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    p0 = [spect.max(), f[spect.argmax()], 512.0]
    bounds = ([1.0, -2e3, 30.0], [np.inf, 2e3, 2e5])

    try:
        popt, pcov = curve_fit(
            model_gaussian, f, spect, p0=p0, bounds=bounds, maxfev=1e5
        )
    except (RuntimeError, ValueError):
        return 0, f_id.size

    A, x0, sigma = popt
    f_min = int(round(x0 - abs(sigma) * np.sqrt(2 * np.log(A / 0.3))))
    f_max = int(round(x0 + abs(sigma) * np.sqrt(2 * np.log(A / 0.3))))
    f_id = f_id.astype(int)
    peak_f0 = np.abs(f_id - f_min).argmin()
    peak_f1 = np.abs(f_id - f_max).argmin() + 1

    if diagnostic_plots:
        from baseband_analysis.utilities import plotting

        if isinstance(diagnostic_plots, bool):
            save_dir = None
        else:
            save_dir = diagnostic_plots

        plotting.spectrum_lim_diagnostic(
            f, spect, f_min, f_max, popt, save_dir=save_dir
        )
    if model_gaussian(f, *popt).max() < min_snr:
        # Does not consider detections smaller than min_snr sigmas
        return 0, f_id.size
    else:
        return peak_f0, peak_f1


def get_weights(
    power_org, f_id, diagnostic_plots=False, spectrum_lim=True, spectrum_thresh=2
):
    """
    Get inverse variance weights for each channel from noise estimate.

    Parameters
    ----------
    power_org: ndarray
        Input power.
    f_id: int
        Frequency index to use.
    diagnostic_plots: bool
        Also produce calibration plots (default False)
    spectrum_lim: bool
        Calculate spectrum limit and restrict to that (default True).

    Returns
    -------
    offset: ndarray
        Means of power.
    weight: ndarray
        Inverse stddev of noise.
    """
    if len(power_org.shape) == 1:
        new_shape = 1
        spectrum_lim = False
    elif len(power_org.shape) == 2:
        new_shape = f_id.size
    elif len(power_org.shape) == 3:
        new_shape = [f_id.size, power_org.shape[1]]

    # Initial rescaling
    power = power_org.copy()
    mean = np.nanmedian(power, axis=-1)
    power -= mean[..., np.newaxis]
    std = np.nanstd(power, axis=-1)
    power /= std[..., np.newaxis]

    # First refinement
    idx_floor = get_floor(power)
    floor = power_org.copy()
    floor[..., idx_floor] = np.nan
    mean = np.nanmean(floor, axis=-1)
    floor -= mean[..., np.newaxis]
    floor[np.isinf(floor)] = np.nan
    std = np.nanstd(floor, axis=-1)

    power = power_org.copy()
    power -= mean[..., np.newaxis]
    power /= std[..., np.newaxis]

    # Second refinement
    idx_floor = get_floor(power)
    floor = power_org.copy()
    floor[..., idx_floor] = np.nan
    mean = np.nanmean(floor, axis=-1)
    floor -= mean[..., np.newaxis]
    floor[np.isinf(floor)] = np.nan
    std = np.nanstd(floor, axis=-1)

    power = power_org.copy()
    power -= mean[..., np.newaxis]
    power /= std[..., np.newaxis]

    # Fit for signal with limited bandwidth
    if spectrum_lim:
        f_lim = get_spectrum_lim(
            f_id, power, diagnostic_plots=diagnostic_plots, min_snr=spectrum_thresh
        )
        if f_lim[1] - f_lim[0] < 20:
            # Ignore if less than 20 channels are left
            log.warning("Less than 20 channels left after spectrum fit.")
        else:
            f_mask = np.nan + np.zeros(new_shape, dtype=float)
            f_mask[f_lim[0] : f_lim[1]] = 0
            mean += f_mask
            std += f_mask

    offset = mean
    weight = np.array(1.0 / std)
    weight[~np.isfinite(weight)] = 0
    return offset, weight


def add_noise(power):
    """
    Substitute NaN values with noise. Noise is taken from random pixels in the off-pulse region.
    While noise is taken independently for each beam, it is mixed across different frequencies.

    Parameters
    ----------
    power : array_like
        Power of data
    freq_id : array_like
        Array of frequency numbers i.e. 1 ~ 800 MHz, 1024 ~ 400 MHz
    downsampling_factor : float
        Additional downsampling factor (excluding the downsampling factor of h5 file).
    mask : array_like, optional
        Mask of missing data. If not specified, calculated using get_mask
    diagnostic_plots : bool, optional
        If True, shows a diagnostic plot. The diagnostic plot shows the mask (where noise is added) in yellow and untouched signal in purple.
    Returns
    -------
    power : array_like
        Power with all masked signal filled with Gaussian noise. If power has many beams, note that the same mask is applied on all beams.
    """
    # Returns if not NaNs are present
    if not np.isnan(power).any():
        return
    # Reduce dimensions for multiple beams
    power_2d = power.copy()
    if len(power.shape) > 2:
        power_2d = power_2d.mean(axis=1)
    # Take noise only from off-pulse waterfall
    floor_idx = ~get_floor(power_2d)
    nan_mask = np.isnan(power_2d)
    noise_idx = np.logical_and(
        ~nan_mask, np.repeat(floor_idx[np.newaxis], power_2d.shape[0], axis=0)
    )
    m = np.arange(power_2d.size).reshape(power_2d.shape)[noise_idx]
    nan_pixels = nan_mask[nan_mask].size
    random = np.random.default_rng(seed=10)
    noise_mask = random.choice(m, size=nan_pixels, replace=True)
    if len(power.shape) == 2:
        power[nan_mask] = power.flatten()[noise_mask]
    else:
        for jj in range(power.shape[1]):
            power[:, jj][nan_mask] = power[:, jj].flatten()[noise_mask]
    return


def normalize(
    data,
    DM=None,
    downsample=False,
    refine_RFI=False,
    valid_channels=None,
    search_valid_channels=True,
    time_range=None,
    thres_mean=5,
    thres_std=3,
    spectrum_lim=False,
    spectrum_thresh=2,
    mask=None,
    w=None,
    diagnostic_plots=False,
    doublecheck_RFI=True,
    check_old_process=True,
    fill_nan=True,
):
    """Normalize power.
    Gets power, applies valid_time_range, sets RFI channels to nan, applies weights and offset. Returns normalized power (3D array) and valid channels.

    Parameters
    ----------
    data : BBD data object
        BBD data object using BBData.from_file(path_to_h5_data_file)
    DM : float, optional
        Dispersion measure for the event in pc/cm^3
    downsample : int or bool, optional
        If int: downsampling factor, if bool: True: downsample down to intensity data resolution
    refine_RFI : bool, optional
        If True, refine RFI using get_RFI_spatial
    valid_channels: 1D ndarray, optional
        Array of channels without RFI. Must have the same size as power.shape[0].
    search_valid_channels : bool, optional
        If True, channels are searched for RFI.
    time_range : array_like, optional
        Index of first and last time bin to keep in the analysis
    thres_mean : float, optional
        Threshold on mean for RFI removal passed to get_RFI_channels. High threshold means few channels will be cut out, low threshold means a lot of channels will be cut out. Should be at least 1.
    thres_std : float, optional
        Threshold on standard deviation for RFI removal passed to get_RFI_channels. High threshold means few channels will be cut out, low threshold means a lot of channels will be cut out. Should be at least 1.
    spectrum_lim : bool, optional
        If True, only keep the frequencies with burst signal. Careful to not use this for very faint bursts.
    mask : array_like, optional
        If provided, the mask is applied before get_RFI_channels. This is so that we apply get_RFI_channels on the floor only (exclude the burst).
    w : list of two 1D ndarray, optional
        Weights and offset of the power. Each one must have the same size as power[valid_channels].shape[0].
    diagnostic_plots : bool, optional
        If True, plot intermediate plots
    check_old_process: bool
        Remove junk data in old processings
    fill_nan : bool, optional
        If NaN values are present, replace them with noise.

    Raises
    ------
    ValueError
        Try a smaller downsampling factor.

    Returns
    -------

    power : array_like
        Normalized power of same shape as power read from input BBData object
    valid_channels : array_like
        Array of 'good' channels.
    """
    # Obtain power
    if DM is None:
        DM = data["tiedbeam_power"].attrs["DM_coherent"]
    freq, freq_id, power = get_power(
        data, DM, diagnostic_plots=diagnostic_plots, check_old_process=check_old_process
    )

    # Define valid channels that are not masked
    if valid_channels is None:
        valid_channels = np.ones_like(freq_id, dtype=bool)
    else:
        valid_channels = valid_channels.copy()
        power = power[valid_channels]
        freq = freq[valid_channels]
        freq_id = freq_id[valid_channels]

    # Mask LTE band RFI
    if search_valid_channels:
        idx_LTE = (freq > 730) & (freq < 760)
        power = power[~idx_LTE]
        freq = freq[~idx_LTE]
        freq_id = freq_id[~idx_LTE]
        valid_channels[valid_channels] = ~idx_LTE

    # Valid tame range
    if time_range is None:
        time_range = get_valid_time_range(power, diagnostic_plots=diagnostic_plots)

    if time_range[1] - time_range[0] == 0:
        power = power[..., 0]
    else:
        power = power[..., time_range[0] : time_range[1]]

    # Downsample
    # With downsample, time_range is in original bins
    # Without downsample, time_range is in power bins
    downsampling_factor = data["tiedbeam_power"].attrs["time_downsample_factor"]
    if type(downsample) is int:
        power, additional_downsample = downsample_power(power, data, factor=downsample)
        downsampling_factor *= downsample
        if power.size == 0:
            raise ValueError("Downsampling factor is too great")
    elif type(downsample) is bool:
        if downsample:
            power, additional_downsample = downsample_power(power, data)
            downsampling_factor *= additional_downsample

    # Substitute NaN values
    idx_nan = np.isnan(power)

    # RFI
    # Apply mask
    if mask is not None:
        floor = power[..., mask]
        RFI_floor = get_RFI_channels(
            floor,
            thres_mean=thres_mean,
            thres_std=thres_std,
            diagnostic_plots=diagnostic_plots,
        )
        power = power[~RFI_floor]
        freq = freq[~RFI_floor]
        freq_id = freq_id[~RFI_floor]
        idx_nan = idx_nan[~RFI_floor]
        valid_channels[valid_channels] = ~RFI_floor
    elif search_valid_channels:
        channels_RFI = get_RFI_channels(
            power,
            thres_mean=thres_mean,
            thres_std=thres_std,
            diagnostic_plots=diagnostic_plots,
        )
        power = power[~channels_RFI]
        freq = freq[~channels_RFI]
        freq_id = freq_id[~channels_RFI]
        idx_nan = idx_nan[~channels_RFI]
        valid_channels[valid_channels] = ~channels_RFI

    # Calculate S/N
    if w is None:
        offset, weight = get_weights(
            power,
            freq_id,
            diagnostic_plots=diagnostic_plots,
            spectrum_lim=spectrum_lim,
            spectrum_thresh=spectrum_thresh,
        )
    else:
        offset, weight = w
        if offset.shape != weight.shape:
            raise IndexError(f"The two elements of w must have the same shape")
        if power.shape[0] != offset.shape[0]:
            raise IndexError(
                f"w has a size of {offset.shape[0]} but {power.shape[0]} is expected"
            )
    power -= offset[..., np.newaxis]
    power *= weight[..., np.newaxis]
    if len(offset.shape) > 1:
        spect_idx = ~np.isnan(offset.sum(axis=1))
    else:
        spect_idx = ~np.isnan(offset)
    power = power[spect_idx]
    freq = freq[spect_idx]
    freq_id = freq_id[spect_idx]
    idx_nan = idx_nan[spect_idx]
    offset = offset[spect_idx]
    weight = weight[spect_idx]
    # Update valid channels with spectrum limits
    valid_channels[valid_channels] = spect_idx

    if refine_RFI:
        # Remove channels highly variable in space
        if len(power.squeeze().shape) < 3:
            log.warning("There must be multiple beams to refine the RFI search")
        else:
            RFI_spacial = get_RFI_spacial(power)
            power = power[~RFI_spacial]
            freq = freq[~RFI_spacial]
            freq_id = freq_id[~RFI_spacial]
            weight = weight[~RFI_spacial]
            offset = offset[~RFI_spacial]
            valid_channels[valid_channels] = ~RFI_spacial

    if doublecheck_RFI:
        # Check RFI again
        idx_floor = ~get_floor(power)
        floor = power[..., idx_floor]
        channels_RFI = get_RFI_channels(floor, diagnostic_plots=diagnostic_plots)
        power = power[~channels_RFI]
        freq = freq[~channels_RFI]
        freq_id = freq_id[~channels_RFI]
        weight = weight[~channels_RFI]
        offset = offset[~channels_RFI]
        valid_channels[valid_channels] = ~channels_RFI

    # Substitute NaN values
    if fill_nan and np.isnan(power).any():
        add_noise(power)

    return (
        power,
        valid_channels,
        offset,
        weight,
        freq,
        freq_id,
        downsampling_factor,
        time_range,
    )


def normalize_and_refine(
    data,
    DM=None,
    downsample=False,
    refine_RFI=False,
    valid_channels=None,
    time_range=None,
    thres_mean=5,
    thres_std=3,
    spectrum_lim=False,
    spectrum_thresh=2,
    mask=None,
    w=None,
    diagnostic_plots=False,
    DM_range=10,
    doublecheck_RFI=True,
    DM_step=0.01,
    check_old_process=True,
    fill_nan=True,
):
    """Wrapper for `normalize`

    Parameters
    ----------
    data : BBD data object
        BBD data object using BBData.from_file(path_to_h5_data_file)
    DM : float, optional
        Dispersion measure for the event in pc/cm^3
    downsample : int or bool, optional
        If int: downsampling factor, if bool: True: downsample down to intensity data resolution
    refine_RFI : bool, optional
        if True, refine RFI using get_RFI_spatial
    valid_channels: array_like, optional
        Array of valid_channels
    time_range : array_like, optional
        Index of first and last time bin to keep in the analysis
    thres_mean : float, optional
        Threshold on mean for RFI removal passed to get_RFI_channels. High threshold means few channels will be cut out, low threshold means a lot of channels will be cut out. Should be at least 1.
    thres_std : float, optional
        Threshold on standard deviation for RFI removal passed to get_RFI_channels. High threshold means few channels will be cut out, low threshold means a lot of channels will be cut out. Should be at least 1.
    spectrum_lim : bool, optional
        If True, only keep the frequencies with burst signal. Careful to not use this for very faint bursts.
    mask : array_like, optional
        If provided, the mask is applied before get_RFI_channels. This is so that we apply get_RFI_channels on the floor only (exclude the burst).
    w : array_like, optional
        Weights and offset of the power
    diagnostic_plots : bool, optional
        If True, plot intermediate plots
    check_old_process: bool
        Remove junk data in old processings
    fill_nan : bool, optional
        If NaN values are present, replace them with noise.

    Returns
    -------
    power : array_like
        Normalized power of same shape as power read from input BBData object
    valid_channels : array_like
        Array of 'good' channels.
    """

    # First normalization
    if DM_range is not None:
        spectrum_lim_auto = False
    else:
        spectrum_lim_auto = spectrum_lim
    (
        power,
        valid_channels_bad,
        offset_bad,
        weight_bad,
        freq,
        freq_id,
        downsampling_factor,
        _,
    ) = normalize(
        data,
        thres_mean=thres_mean,
        thres_std=thres_std,
        w=w,
        downsample=downsample,
        time_range=time_range,
        valid_channels=valid_channels,
        spectrum_lim=spectrum_lim_auto,
        DM=DM,
        check_old_process=check_old_process,
        doublecheck_RFI=doublecheck_RFI,
    )

    # DM refinement
    if DM is None:
        DM = data["tiedbeam_power"].attrs["DM_coherent"]
    if DM_range is not None:
        DM = DM_SNR(
            power.copy(),
            freq,
            DM,
            DM_range,
            DM_step,
            dt=data.attrs["delta_time"] * downsampling_factor,
            diagnostic_plots=diagnostic_plots,
        )
        log.info(f"Best DM found at {DM} pc/cc")
        (
            power,
            valid_channels_bad,
            offset_bad,
            weight_bad,
            freq,
            freq_id,
            downsampling_factor,
            _,
        ) = normalize(
            data,
            thres_mean=thres_mean,
            thres_std=thres_std,
            w=w,
            downsample=downsample,
            time_range=time_range,
            valid_channels=valid_channels,
            spectrum_lim=spectrum_lim,
            DM=DM,
            check_old_process=check_old_process,
            doublecheck_RFI=doublecheck_RFI,
        )

    idx_floor = ~get_floor(power)
    (
        power,
        valid_channels,
        offset,
        weight,
        freq,
        freq_id,
        downsampling_factor,
        time_range_out,
    ) = normalize(
        data,
        DM=DM,
        downsample=downsample,
        refine_RFI=refine_RFI,
        w=w,
        time_range=time_range,
        valid_channels=valid_channels,
        spectrum_lim=spectrum_lim,
        spectrum_thresh=spectrum_thresh,
        thres_mean=thres_mean,
        thres_std=thres_std,
        mask=idx_floor,
        doublecheck_RFI=doublecheck_RFI,
        check_old_process=check_old_process,
        fill_nan=fill_nan,
    )

    return (
        power,
        valid_channels,
        offset,
        weight,
        freq,
        freq_id,
        downsampling_factor,
        time_range_out,
        DM,
    )


def tiedbeam_baseband_to_power(
    data, time_downsample_factor=1, dm=None, dedisperse=True, time_shift=False
):
    """Compute power in tiedbeam baseband data (in-place)."""
    ntime = data.ntime - (data.ntime % time_downsample_factor)
    ntime_downsampled = ntime // time_downsample_factor

    if dedisperse:
        if dm is None:
            try:
                dm = data["tiedbeam_baseband"].attrs["DM"]
            except KeyError:
                measured_parameter = get_measured_parameter(data.attrs["event_id"])
                dm = measured_parameter["dm"]
        bb = coherent_dedisp(data, dm, time_shift=time_shift)[..., :ntime]
    else:
        if dm is None:
            dm = 0.0
        bb = data["tiedbeam_baseband"][..., :ntime]

    power = np.abs(bb) ** 2
    s = power.shape
    power.shape = s[:-1] + (ntime_downsampled, time_downsample_factor)
    power = np.nanmean(power, -1)

    pow_dset = data.create_dataset("tiedbeam_power", data=power)
    pow_dset.attrs["axis"] = list(
        data["tiedbeam_baseband"].attrs["axis"]
    )  # ['freq','beam','time']
    pow_dset.attrs["fill_value"] = np.float32(np.nan)
    pow_dset.attrs["time_downsample_factor"] = time_downsample_factor
    pow_dset.attrs["DM_coherent"] = dm
    try:
        pow_dset.attrs["calibrator"] = data["tiedbeam_baseband"].attrs["calibrator"]
    except KeyError:
        pass

    time_map = data.index_map["time"][:ntime]
    time_map.shape = (ntime_downsampled, time_downsample_factor)

    ds_time_map = np.empty(
        ntime_downsampled, dtype=[("offset_fpga", float), ("offset_s", np.float64)]
    )
    ds_time_map["offset_fpga"] = np.mean(time_map["offset_fpga"], -1)
    ds_time_map["offset_s"] = np.mean(time_map["offset_s"], -1)
    data.create_index_map("downsampled_time", ds_time_map)
    return