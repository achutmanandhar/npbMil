function sortObj = vbMilSortCount(counts)

    sortObj.unsortedCounts = counts;
    [sortObj.sortedCounts, sortObj.sortIndices] = sort(counts,2,'descend');
    [~, sortObj.unsortIndices] = sort(sortObj.sortIndices,2,'ascend');

end