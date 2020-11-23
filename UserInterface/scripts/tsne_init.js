

function initialize_tsne_view (dataset = ORIGINAL_DATASET) {

    show_view("tsne");

    $TSNE.empty();

	// SS = init_tsne(dataset)


    // init global SVG and MARGIN
    TSNE_MARGIN = {top: 10, right: 60, bottom: 100, left: 0};

    TSNE_SVG = d3.select("#tsne-svg-container").append("svg")
        .attr("id", "tsne-svg")
        .attr("width", $TSNE.width())
        .attr("height", $TSNE.height())
        .append("g")
        .attr("transform", "translate(" + TSNE_MARGIN.left + "," + TSNE_MARGIN.top + ")");



    SS = init_tsne(dataset);

}


function init_tsne (dataset) {

    var data_tsne = dataset;
    pn = [];
    for (var j = 0; j < data_tsne.length; j ++) {
    	pn[j] = data_tsne[j]["Patient"];
    }


    var xx = [];
    var yy = [];
    for (var j = 0; j < dataset.length; j ++) {
            xx[j] = dataset[j]["x"] 
            yy[j] = dataset[j]["y"] 
        };


   	console.log('t-SNE Correct!')

    colors1 = Array(ORIGINAL_DATASET.length).fill('#00304e');
    var update1 = {'marker':{color: colors1, size:10}};

    var trace = {
      x: xx,
      y: yy,
      mode: 'markers',
      type: 'scatter',
      hoverinfo: 'text',
      text: pn,
      marker: { size: 8, color: colors1}
    };
	var dataaa = [trace];
	var layout = {
      // width: 500,
      height: 350,
      xaxis: {
        autorange: true,
        showgrid: false,
        zeroline: false,
        showline: false,
        autotick: false,
        showticklabels: false,
      },
      yaxis: {
        autorange: true,
        showgrid: false,
        zeroline: false,
        showline: false,
        autotick: false,
        showticklabels: false,
      },
      title:'<b>t-SNE Plot</b>',
      titlefont: {
        // family: 'Arial, sans-serif',
        // family: 'Titillium',
        family: 'Titillium Web', 
        size: 18,
        color: 'black'
      },
    };
	Plotly.newPlot('tsne-svg-container', dataaa, layout, {showSendToCloud: true, scrollZoom: true});


    var myPlot = document.getElementById('tsne-svg-container');
    myPlot.on('plotly_click', function(data){
        var pn = data.points[0].pointNumber,
        colors2 = Array(ORIGINAL_DATASET.length).fill('#00304e');    
        colors2[pn] = '#ffc000';
        // console.log(data.points[0].text)
        // console.log(pn)
        // console.log(data.points[0].text)
        var u1 = {'marker':{color: colors1, size:10}};
        var update2 = {'marker':{color: colors2, size:10}};
        Plotly.restyle('tsne-svg-container', u1);
        Plotly.restyle('tsne-svg-container', update2);
        
        enter_select_mode(data.points[0].text, true);

    });

    return {
        fourth: update1,
    };
}








function enter_select_tsne_view (case_name) {
    exit_select_tsne_view();

   var myPlott = document.getElementById('tsne-svg-container');
    var datagraph = myPlott.data;

    for (var j = 0; j < datagraph[0].text.length; j ++) {
        if (datagraph[0].text[j] == case_name) {
                var test_value = 1;
                    colors3 = Array(ORIGINAL_DATASET.length).fill('#00304e');
                    colors3[j] = '#ffc000';
                    var update3 = {'marker':{color: colors3, size:10}};
                    Plotly.restyle('tsne-svg-container', update3);
            } else {
                var test_value = 0;
            }
    };



}



function exit_select_tsne_view () {
    Plotly.restyle('tsne-svg-container', SS.fourth);

}




