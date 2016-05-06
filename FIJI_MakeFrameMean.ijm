   for (i=1; i<=nSlices; i++) { 
       	setSlice(i); 
       	getStatistics(area, mean, min, max, std);
       	setColor(mean); 
       	run("Make Inverse");
       	fill(); 
       	run("Make Inverse");
   } 