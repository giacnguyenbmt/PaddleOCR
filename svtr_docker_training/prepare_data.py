import os
import sys


def check_label(label, char_list):
    for char_ in label:
        if char_ not in char_list:
            return False
    return True


def create_label_file(real_dir,
                      synth_dir,
                      dst_file,
                      chars='0123456789ABCDĐEFGHKLMNPQRSTUVXYZ'):
    filenames = []
    labels = []
    fs = []
    for path, subdirs, files in os.walk(real_dir):
        for name in files:
            fs.append(os.path.join(path, name))
    for filename in fs:
        base_name = os.path.basename(filename)
        if "_" in base_name:
            label = base_name.split('_')[0]  # format: [label]_[random number].jpg
        else:
            label = base_name.split('.')[0]
        label = label.upper()
        if check_label(label, chars) is False:
            continue
        labels.append(label)
        filenames.append(filename)
    print("***************************************")
    print("Real data:", len(labels))
    
    # synthysize
    filenames_syn = []
    labels_syn = []
    if synth_dir is not None:
        fs = []
        for path, subdirs, files in os.walk(synth_dir):
            for name in files:
                fs.append(os.path.join(path, name))
        for filename in fs:
            base_name = os.path.basename(filename)
            if "_" in base_name:
                label = base_name.split('_')[0]
            else:
                label = base_name.split('.')[0]
            label = label.upper()
            if check_label(label, chars) is False:
                continue
            labels_syn.append(label)
            filenames_syn.append(filename)
        print("***************************************")
        print("Synth data:", len(labels_syn))

    entire_filenames = filenames + filenames_syn
    entire_labels = labels + labels_syn

    with open(dst_file, 'w') as f:
        for file_path, lp_num in zip(entire_filenames, entire_labels):
            f.write('{}\t{}\n'.format(file_path, lp_num))


if __name__ == '__main__':
    real_dir = sys.argv[1]
    synth_dir = sys.argv[2]
    dst_file = sys.argv[3]
    if synth_dir == 'none':
        synth_dir = None
    create_label_file(real_dir, synth_dir, dst_file)
