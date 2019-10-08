from fiction_segmentation import const
import fiction_segmentation.preprocessing as P


def split_overflow(sent, prev_len, sample_size=const.SAMPLE_SIZE):
    first, rest = sent[0: sample_size - prev_len], sent[sample_size - prev_len:]
    *ready, last = [rest[i:i + sample_size] for i in range(0, len(rest), sample_size)]
    return first, ready, last


def gen_samples(content, sample_size=const.SAMPLE_SIZE):
    samples = []
    sample = []
    for sentence in content:
        if len(sample) + len(sentence) <= sample_size:
            sample.extend(sentence)
        else:
            if len(sentence) <= sample_size:
                last = sentence
            else:
                first, ready, last = split_overflow(sentence, sample_size, len(sample))
                sample.extend(first)
                samples.append(sample)
                samples.extend(ready)
            sample = last
            samples.append(P.pad_sequence(sample, sample_size))
    return samples


def gen_data(book_dir, w2v_data_path=const.W2V_DATA_PATH, sample_size=const.SAMPLE_SIZE):
    # TODO implement gen_data

    return


def one_hot_encode(classes, num_classes=const.NUM_CLASSES):
    # TODO implement one_hot_encode
    return


def write_dataset():
    # TODO implement write_dataset
    return
