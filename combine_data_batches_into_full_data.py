import pickle
import bz2

def main():
    full_x_train = []
    full_y_train = []
    full_x_test = []
    full_y_test = []


    for i in range(8):
        file_name = "generated_data" + str(i) + ".bz2"
        in_file = bz2.BZ2File(file_name,'rb')
        data = pickle.load(in_file)
        in_file.close()

        full_x_train.extend(data.x_train)
        full_y_train.extend(data.y_train)
        full_x_test.extend(data.x_test)
        full_y_test.extend(data.y_test)

    data_tup = (full_x_train,full_y_train,full_x_test,full_y_test)

    o_file = "full_generated_data.bz2"
    out = bz2.BZ2File(o_file,'wb')
    pickle.dump(data_tup,out)
    out.close()

if __name__ == "__main__":
    main()
