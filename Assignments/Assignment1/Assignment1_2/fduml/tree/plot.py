# vis the decision tree
# ref：https://github.com/JonathonYan1993/ML_DecisionTree_prepruning_postpruning

import matplotlib.pyplot as plt

# support Chinese
plt.rcParams['font.sans-serif'] = ['SimHei']

# node boxstyle
nonleafNode = dict(boxstyle="round", facecolor="white", mutation_scale="1.2", ls="--")
leafNode = dict(boxstyle="square", mutation_scale="1.2")
arrow_args = dict(arrowstyle="<-")

# https://matplotlib.org/stable/gallery/color/named_colors.html
colors = ['peachpuff', 'yellowgreen', 'palevioletred', 'skyblue', 'darkorange', 'blueviolet', 'slategrep', 'khaki', 'silver', 'teal']

def plotNode(nodeTxt, centerPt, parentPt, nodeType, ax):
    """
    plot node
    :param nodeTxt: text on the node
    :param centerPt: position center of the node
    :param parentPt: end of the arrow
    :param nodeType: node type
    :param ax: figure
    :return:
    """
    ax.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                xytext=centerPt, textcoords='axes fraction',
                size='large',
                va="bottom", ha="center",
                bbox=nodeType, arrowprops=arrow_args)


def plotSubTree(clf, tmpNode, parentPt, ax, xOff, yOff, featNames, classNames):
    """
    plot subtree
    :param clf: clf
    :param tmpNode: tmp node
    :param parentPt: coordinate of the parent node
    :param ax: figure
    :param xOff: initial X-axis offset
    :param yOff: initial y-axis offset
    :param featNames: feature names
    :param classNames: class names
    :return:
    """
    numLeafs = tmpNode.leaf_num

    featStr = 'feat_id: ' + str(tmpNode.feat_idx)
    thresStr = 'threshold: ' + str(tmpNode.threshold)
    classStr = 'class: ' + str(tmpNode.value)
    if len(featNames) != 0 and tmpNode.feat_idx != None:
        featStr = "feat：" + featNames[tmpNode.feat_idx]
    if len(classNames) != 0 and tmpNode.value != None:
        classStr = 'class: ' + classNames[tmpNode.value]

    tmpxOff = (clf.tree_leaf_num / 3.0) * (xOff + numLeafs / 2.0 / float(clf.tree_leaf_num))
    cntrPt = (tmpxOff, (1.0 + clf.tree_leaf_num / 25.0) * yOff)

    if parentPt == (0, 0):
        parentPt = cntrPt

    if tmpNode.left == None and tmpNode.right == None:
        nodeStr = "\n" + classStr + "\n"
        leafNode["fc"] = colors[tmpNode.value]
        plotNode(nodeStr, cntrPt, parentPt, leafNode, ax)
    else:
        nodeStr = featStr + "\n" + thresStr + "\n" + classStr
        plotNode(nodeStr, cntrPt, parentPt, nonleafNode, ax)

    yOff = yOff - 1.0 / float(clf.tree_depth)
    if tmpNode.left != None:
        plotSubTree(clf, tmpNode.left, cntrPt, ax, xOff, yOff, featNames, classNames)
    if tmpNode.right != None:
        xOff = xOff + float(tmpNode.left.leaf_num) / float(clf.tree_leaf_num)
        plotSubTree(clf, tmpNode.right, cntrPt, ax, xOff, yOff, featNames, classNames)


def plot_tree(clf, featNames, classNames):
    """
    main function
    :param clf: decision tree classifier
    :param featNames: feature names, a python list
    :param classNames: class names, a python list
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])

    ax = plt.subplot(111, frameon=False, **axprops)
    xOff = -0.5 / float(clf.tree_leaf_num)
    yOff = 1.0

    plotSubTree(clf, clf.root, (0, 0), ax, xOff, yOff, featNames, classNames)
