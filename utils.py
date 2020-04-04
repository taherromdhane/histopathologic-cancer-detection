import os

def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tif', '')
    
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))