
import os

items = []

with open("log.txt", "r") as logfile:
    item = {}
    for line in logfile.readlines():
        if '[Current Config Option]' in line:
            i = line.find("\"axis_0\": [") + 11
            j = line.find("], \"reorder\"")
            dims = [int(k) for k in line[i:j].split(",")]
            item = [dims]
        if 'dram_read_bytes' in line:
            dram_read = float(line.split(" ")[-1])
            item.append(int(dram_read))
        if 'dram_write_bytes' in line:
            dram_write = int(line.split(" ")[-1])
            item.append(dram_write)
        if 'Elapsed Cycles' in line:
            cycles_str = line.split(" ")[-1]
            cycles = int(cycles_str.replace(',', ''))
            item.append(cycles)
        if 'Memory Throughput' in line:
            tp = float(line.split(" ")[-1])
            item.append(tp)
        if "L1/TEX Hit Rate" in line:
            l1_hit = float(line.split(" ")[-1])
            item.append(l1_hit)
        if "L2 Hit Rate" in line:
            l2_hit = float(line.split(" ")[-1])
            item.append(l2_hit)
            items.append(item)


threads = [8, 32, 128]

for thread in threads:
    with open("dram_read_{}.csv".format(thread), "w") as f:
        for item in items:
            if item[0][2] == thread:
                if item[0][3] < 256:
                    f.write(str(item[1]) + ",")
                else:
                    f.write(str(item[1]) + "\n")

    with open("dram_write_{}.csv".format(thread), "w") as f:
        for item in items:
            if item[0][2] == thread:
                if item[0][3] < 256:
                    f.write(str(item[2]) + ",")
                else:
                    f.write(str(item[2]) + "\n")

    with open("cycles_{}.csv".format(thread), "w") as f:
        for item in items:
            if item[0][2] == thread:
                if item[0][3] < 256:
                    f.write(str(item[3]) + ",")
                else:
                    f.write(str(item[3]) + "\n")

    with open("memory_throughput_{}.csv".format(thread), "w") as f:
        for item in items:
            if item[0][2] == thread:
                if item[0][3] < 256:
                    f.write(str(item[4]) + ",")
                else:
                    f.write(str(item[4]) + "\n")

    with open("l1_hit_{}.csv".format(thread), "w") as f:
        for item in items:
            if item[0][2] == thread:
                if item[0][3] < 256:
                    f.write(str(item[5]) + ",")
                else:
                    f.write(str(item[5]) + "\n")

    with open("l2_hit_{}.csv".format(thread), "w") as f:
        for item in items:
            if item[0][2] == thread:
                if item[0][3] < 256:
                    f.write(str(item[6]) + ",")
                else:
                    f.write(str(item[6]) + "\n")

