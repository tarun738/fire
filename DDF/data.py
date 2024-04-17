import glob
import logging
import numpy as np
import os
import torch
import torch.utils.data
import DDF.workspace as ws
import trimesh as tm

def get_instance_filenames(data_source, split, appendName, type=None):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                datasetName = dataset
                # datasetName = dataset[:-2]
                instance_filename = os.path.join(
                    datasetName, class_name, instance_name + appendName + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ) and not os.path.isdir(os.path.join(data_source, instance_filename[:-4])):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    if type == "mesh":
                        logging.warning(
                            "Requested non-existent file '{}'".format(instance_filename)
                        )
                    else:
                        logging.info("Requested non-existent file '{}' continuing anyway".format(instance_filename))
                        npzfiles += [instance_filename]
                else:
                    npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        return NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename,missingDirs=False):
    npz = np.load(filename, allow_pickle=True)
    samples_dsdf = torch.from_numpy(npz["dsdf"]).float()
    samples_pc = samples_dsdf[:,0:3] + samples_dsdf[:,3:6] *  samples_dsdf[:,6].unsqueeze(-1)
    if "sdf" in npz:
        samples_sdf = npz["sdf"][None][0]
        samples_sdf_pos = torch.from_numpy(samples_sdf["pos"]).float()
        samples_sdf_neg = torch.from_numpy(samples_sdf["neg"]).float()
    else:
        samples_sdf_pos = []
        samples_sdf_neg = []
    retData = [samples_dsdf,samples_pc, samples_sdf_pos,samples_sdf_neg]
    
    if missingDirs:
        npz = np.load(filename[:-4]+"_misses.npz", allow_pickle=True)
        samples_dsdf_miss = torch.from_numpy(npz["dsdfmiss"]).float()
        retData.append(samples_dsdf_miss)
    return retData

def unpack_sdf_samples(filename, subsample=None, missingDirs=False):
    data = read_sdf_samples_into_ram(filename,missingDirs=missingDirs)
    if subsample is None:
        return data
    return unpack_sdf_samples_from_ram(data,subsample, missingDirs=missingDirs)


def unpack_sdf_samples_from_ram(data, subsample=None, missingDirs=False):
    if subsample is None:
        return data[0]
    samples_dsdf = data[0]
    if missingDirs:
        samples_dsdf_missing = data[-1]
    if not missingDirs:
        random = (torch.rand(subsample) * samples_dsdf.shape[0]).long()
        samples_dsdf = torch.index_select(samples_dsdf, 0, random)
        samples_dsdf = torch.cat([samples_dsdf,torch.ones([samples_dsdf.shape[0],1])],dim=-1)
    else:
        random1 = (torch.rand(int(subsample//4)) * samples_dsdf_missing.shape[0]).long()
        random2 = (torch.rand(int((subsample*3)//4)) * samples_dsdf.shape[0]).long()
        samples_dsdf = torch.index_select(samples_dsdf, 0, random2)
        samples_dsdf = torch.cat([samples_dsdf,torch.ones([samples_dsdf.shape[0],1])],dim=-1)
        samples_dsdf_missing = torch.index_select(samples_dsdf_missing, 0, random1)
        samples_dsdf_missing = torch.cat([samples_dsdf_missing,torch.zeros([samples_dsdf_missing.shape[0],1])],dim=-1)
        samples_dsdf = torch.cat([samples_dsdf,samples_dsdf_missing],dim=0)

    samples_sdf_pos = data[2]
    samples_sdf_neg = data[3]
    half = int(subsample/2)
    random = (torch.rand(half) * samples_sdf_pos.shape[0]).long()
    samples_sdf_pos = torch.index_select(samples_sdf_pos, 0, random)
    samples_sdf_pos = samples_sdf_pos.float()
    random = (torch.rand(half) * samples_sdf_neg.shape[0]).long()
    samples_sdf_neg = torch.index_select(samples_sdf_neg, 0, random)
    samples_sdf_neg = samples_sdf_neg.float()

    samples_sdf = torch.cat([samples_sdf_pos,samples_sdf_neg],dim=0)

    return torch.cat([samples_dsdf,samples_sdf],dim=1)


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        appendName,
        load_ram=False,
        missingDirs=False,
        numDataSamplesPerScene=-1,
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split, appendName)
        self.num_pc = 1000000
        self.missingDirs = missingDirs
        logging.info(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )
        self.load_ram = load_ram
        if load_ram:
            self.loaded_data = []
            fileNum=0
            del_files = []
            for f in self.npyfiles:
                if not os.path.isfile(os.path.join(self.data_source, ws.sdf_samples_subdir, f)):
                    del_files.append(f)
                    continue
                logging.info("Loaded file number {}".format(fileNum))
                fileNum+=1
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename, allow_pickle=True)
                samples_dsdf = remove_nans(torch.from_numpy(npz["dsdf"]))
                samples_dsdf = samples_dsdf[torch.randperm(samples_dsdf.shape[0])]
                samples_dsdf = samples_dsdf[:numDataSamplesPerScene,:]
                logging.info("Number of samples for DDF: {}".format(samples_dsdf.shape))

                if self.missingDirs:
                    filename_dsdf_miss = os.path.join(self.data_source, ws.sdf_samples_subdir, f[:-4]+"_misses.npz")
                    npz_miss = np.load(filename_dsdf_miss, allow_pickle=True)
                    samples_dsdf_misses = remove_nans(torch.from_numpy(npz_miss["dsdfmiss"]))
                    samples_dsdf_misses = samples_dsdf_misses[torch.randperm(samples_dsdf_misses.shape[0])]
                    samples_dsdf_misses = samples_dsdf_misses[:numDataSamplesPerScene,:]
                    logging.info("Number of missing samples for DDF: {}".format(samples_dsdf_misses.shape))
                currData =  [
                        samples_dsdf[torch.randperm(samples_dsdf.shape[0])],
                        torch.zeros((0,0))
                    ]

                samples_sdf = npz["sdf"][None][0]
                samples_sdf_pos = remove_nans(torch.from_numpy(samples_sdf["pos"]))
                samples_sdf_neg = remove_nans(torch.from_numpy(samples_sdf["neg"]))
                samples_sdf_pos = samples_sdf_pos[:,:4]
                samples_sdf_neg = samples_sdf_neg[:,:4]
                currData.append(samples_sdf_pos[torch.randperm(samples_sdf_pos.shape[0])])
                currData.append(samples_sdf_neg[torch.randperm(samples_sdf_neg.shape[0])])
                if self.missingDirs:
                    currData.append(samples_dsdf_misses)
                self.loaded_data.append(currData)
            for f in del_files:
                self.npyfiles.remove(f)
            logging.info("Loaded {} scenes".format(len(self.npyfiles)))
    def __len__(self):
        return len(self.npyfiles)
    def getLoadedData(self):
        return self.loaded_data
    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample,missingDirs=self.missingDirs), idx
        else:
            return unpack_sdf_samples(filename, self.subsample,missingDirs=self.missingDirs), idx
