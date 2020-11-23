

function initialize_umap_view (dataset = ORIGINAL_DATASET) {

    show_view("umap");

    $UMAP.empty();



    // init global SVG and MARGIN
    UMAP_MARGIN = {top: 10, right: 60, bottom: 100, left: 0};

    UMAP_SVG = d3.select("#umap-svg-container").append("svg")
        .attr("id", "umap-svg")
        .attr("width", $UMAP.width())
        .attr("height", $UMAP.height())
        .append("g")
        .attr("transform", "translate(" + UMAP_MARGIN.left + "," + UMAP_MARGIN.top + ")");



    SS = init_umap(dataset);

}


function init_umap (dataset) {

    var data_umap = dataset;
    pn = [];
    for (var j = 0; j < data_umap.length; j ++) {
    	pn[j] = data_umap[j]["Patient"];
    }


    var uu = [];
    var vv = [];
    for (var j = 0; j < dataset.length; j ++) {
            uu[j] = dataset[j]["u"] 
            vv[j] = dataset[j]["v"] 
        };


   	console.log('UMAP Correct!')

    colors1 = Array(ORIGINAL_DATASET.length).fill('#00304e');
    var update1 = {'marker':{color: colors1, size:10}};

    var trace = {
      x: uu,
      y: vv,
      mode: 'markers',
      type: 'scatter',
      hoverinfo: 'text',
      text: pn,
      marker: { size: 8, color: colors1}
    };
	var dataaa = [trace];
	var layout = {
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
      title:'<b>UMAP Plot</b>',
      titlefont: {
        // family: 'Arial, sans-serif',
        // family: 'Titillium',
        family: 'Titillium Web',
        size: 18,
        color: 'black'
      },
    };
	Plotly.newPlot('umap-svg-container', dataaa, layout, {showSendToCloud: true, scrollZoom: true});


    var myPlot = document.getElementById('umap-svg-container');
    myPlot.on('plotly_click', function(data){
        var pn = data.points[0].pointNumber,
        colors2 = Array(ORIGINAL_DATASET.length).fill('#00304e');    
        colors2[pn] = '#ffc000';
        var u1 = {'marker':{color: colors1, size:10}};
        var update2 = {'marker':{color: colors2, size:10}};
        Plotly.restyle('umap-svg-container', u1);
        Plotly.restyle('umap-svg-container', update2);
        
        enter_select_mode(data.points[0].text, true);

    });

    return {
        fourth: update1,
    };
}




function enter_select_umap_view (case_name) {
    exit_select_umap_view();

   var myPlott = document.getElementById('umap-svg-container');
    var datagraph = myPlott.data;

    for (var j = 0; j < datagraph[0].text.length; j ++) {
        if (datagraph[0].text[j] == case_name) {
                var test_value = 1;
                    colors3 = Array(ORIGINAL_DATASET.length).fill('#00304e');
                    colors3[j] = '#ffc000';
                    var update3 = {'marker':{color: colors3, size:10}};
                    Plotly.restyle('umap-svg-container', update3);
            } else {
                var test_value = 0;
            }
    };



}



function exit_select_umap_view () {
    Plotly.restyle('umap-svg-container', SS.fourth);

}



