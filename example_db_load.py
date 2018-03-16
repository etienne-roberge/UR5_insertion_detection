import pickle
import matplotlib.pyplot as plt
import random

#Dataset is build like this:
#(1) dataset: Contains every example of the signals saved during the robots teaching, contains 9 signals (125Hz).
        # 1 - X forces(N)
        # 2 - Y forces(N)
        # 3 - Z forces(N)
        # 4 - X Torques(N/m)
        # 5 - Y Torques(N/m)
        # 6 - Z Torques(N/m)
        # 7 - X Tool speed(mm/s)
        # 8 - Y Tool speed(mm/s)
        # 9 - Z Tool speed(mm/s)
#(2) Ground-truth boxes: Contains the index of where every actions starts and ends in every examples
        # ex: [[100,200],[1380,2000]] = 1st actions of the signal starts at index 100 and ends at index 200
#(3) Ground-truth labels: Contains the label of every actions in every examples. The possible actions are:
        # 1 = insertion
        # 2 = placing an object on the table


DB_name = "Insertion_database_16March2018.dataset"

with open(DB_name) as tmp:
    print("Opening dataset")
    dataset, dataset_gtBoxes, dataset_gtBoxes_labels = pickle.load(tmp)


#Let's try to plot the force/speed signals and the ground-truth of a random example
    index = random.randint(0,len(dataset))
    signals = dataset[index]
    gt_boxes = dataset_gtBoxes[index]
    labels = dataset_gtBoxes_labels[index]

    #plot signals
    fig, ax1 = plt.subplots(1, 1)
    for sig in signals:
        ax1.plot(sig)

    #plot gt-boxes
    for i in range(len(gt_boxes)):
        if(labels[i]==1):
            color = "blue"
            name = "Insertion"
        elif(labels[i]==2):
            color = "red"
            name = "Pick-and-place"
        ax1.axvspan(gt_boxes[i][0], gt_boxes[i][1], alpha=0.3, color=color, label=name)

    plt.legend()
    plt.show() 
