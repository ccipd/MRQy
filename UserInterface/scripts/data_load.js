
function data_loading () {

	var $this = $(this);
	var cur_file = null;

	// escape the cancelation case
	if ($this.val() == "") {
		return;
	} else {
		cur_file = $this.get(0).files[0];
	}

	// hide the "Upload Dataset" button
	$("#upload-button").css("display", "none");

	// read dataset from the file
	FILE_NAME = cur_file.name.split(".")[0];
	console.log("[LOG] Read in file: " + FILE_NAME);
	var fileReader = new FileReader();
	fileReader.readAsText(cur_file);
	fileReader.onload = function () {

		console.log("[LOG] App initializing...");
		var file_text = fileReader.result;


		var absdirRe = /#outdir:?\s*([^\s]*)\s*\n/;
		var abs_outdir = absdirRe.exec(file_text)[1];
		var reldirRe = /([^\\\/]*)$/;
		var rel_outdir = reldirRe.exec(abs_outdir)[1];
		DATA_PATH = DATA_PATH + rel_outdir + "/" ;
		FILE_HEADER = file_text.split(/#dataset:\s?/)[0] + "#dataset: ";
		dataset_text = file_text.split(/#dataset:\s?/)[1];
		

		ORIGINAL_DATASET = d3.tsv.parse(dataset_text, function(d) {
			if (d.hasOwnProperty("")) delete d[""];
			for (var key in d) {
			    if ($.isNumeric(d[key])) {
			      d[key] = Math.round(parseFloat(d[key]) * 100) / 100; // Round the value to two digits
			    }
			  }
			  return d;
		});

		
		



		columnInfo = {};

		ORIGINAL_DATASET1 = d3.tsv.parse(dataset_text, function (d) {
			if (d.hasOwnProperty("")) delete d[""];
			for (var key in d) {
				if (d[key] === "") {
					d[key] = "N/A";
					if (!columnInfo.hasOwnProperty(key)) {
						columnInfo[key] = "with N/A content";
					}
				} else if ($.isNumeric(d[key])) {
					d[key] = +d[key];
					if (!columnInfo.hasOwnProperty(key)) {
						columnInfo[key] = "numerical";
					}
				} else {
					if (!columnInfo.hasOwnProperty(key)) {
						columnInfo[key] = "non-numerical";
					}
				}
			}
			return d;
		});

		// Sort the keys by column type (non-numerical first, then numerical)
		// Keep the "participant" key first
		var sortedKeys = Object.keys(columnInfo).sort(function (a, b) {
			if (a === "Participant") return -1;
			if (b === "Participant") return 1;
			return columnInfo[a] === "non-numerical" ? -1 : 1;
		});

		// Update the ORIGINAL_DATASET1 by rearranging the columns based on sortedKeys
		ORIGINAL_DATASET1 = ORIGINAL_DATASET1.map(function (row) {
			var newRow = {};
			sortedKeys.forEach(function (key) {
				newRow[key] = row[key];
			});
			return newRow;
		});



		// console.log(columnInfo)








		ORIGINAL_DATASET3 = d3.tsv.parse(dataset_text, function (d) {
			if (d.hasOwnProperty("")) delete d[""];
			for (var key in d) {
				if ($.isNumeric(d[key])) {
					d[key] = +d[key];
				}
			}
			return d;
		});



		



		for (var j = 0; j < ORIGINAL_DATASET.length; j ++) {
			image_names[j] = ORIGINAL_DATASET[j]["Name of Images"];
			participant_names[j] = ORIGINAL_DATASET[j]["Participant"];
			delete ORIGINAL_DATASET[j]["Name of Images"];
		};





		for (var j = 0; j < ORIGINAL_DATASET1.length; j ++) {
			delete ORIGINAL_DATASET1[j]["Name of Images"];
		};

		for (var j = 0; j < ORIGINAL_DATASET3.length; j ++) {
			delete ORIGINAL_DATASET3[j]["Name of Images"];
		};


		




		// show the current loaded dataset name
		$("#dataset-tag").css("display", "inline")
						 .text("Data: " + cur_file.name + " | N= " + ORIGINAL_DATASET.length + ", " + rel_outdir);
						 // .text("Current dataset: " + cur_file.name + " | Number of Participant: " + (ORIGINAL_DATASET.length - 8)/4  + ", CCF_Crohns_CTEs");



		// build case list.		
		ORIGINAL_CASE_LIST = ORIGINAL_DATASET.map(function(d){return d["Participant"];});

		// build case dict with casename as key. 
		for (var i = 0; i < ORIGINAL_DATASET.length; i ++) {
			var cur_file_name = ORIGINAL_DATASET[i]["Participant"];
			ORIGINAL_CASE_DICT[cur_file_name] = {};
			for (var index in FEATURES_TO_MAP) {
				ORIGINAL_CASE_DICT[cur_file_name][FEATURES_TO_MAP[index]] = ORIGINAL_DATASET[i][FEATURES_TO_MAP[index]];
			}
			ORIGINAL_CASE_DICT[cur_file_name]["dom_id"] = cur_file_name.replace(/\.|\#/g, "-");
		}
		
		var img = new Image();
		img.typeidx = 0;
		img.onerror = (function () {
			CHECK_IMAGE_EXTENSIONS[this.typeidx] = true;
		});
		img.src = "";
	// }





		// build feature list
		ORIGINAL_FEATURE_LIST = Object.keys(ORIGINAL_DATASET[0]);
		// ORIGINAL_FEATURE_LIST2 = Object.keys(ORIGINAL_DATASET2[0]);
		// ORIGINAL_FEATURE_LIST1 = Object.keys(ORIGINAL_DATASET1[0]);

		ORIGINAL_FEATURE_LIST1 = Object.keys(ORIGINAL_DATASET1[0]);
		// console.log(ORIGINAL_FEATURE_LIST1)
		ORIGINAL_FEATURE_LIST3 = Object.keys(ORIGINAL_DATASET3[0]);


		CURRENT_MULTI_SELECTED = ORIGINAL_DATASET;

		var image_check_interval = setInterval (function () {
			var check_sum = 0;
			for (var ck_index = 0; ck_index < CHECK_IMAGE_EXTENSIONS.length; ck_index ++) {
				check_sum += CHECK_IMAGE_EXTENSIONS[ck_index];
			}
			if (check_sum == CHECK_IMAGE_EXTENSIONS.length) {
				clearInterval (image_check_interval);

				// initialize table view
				initialize_data_table(ORIGINAL_DATASET);
				if (!OPEN_WITH_TABLE) {
					hide_view("table");
				}
				d3.select("#table-btn")
					.classed("view-mngmt-btn-hidden", false)
					.classed("view-enabled", OPEN_WITH_TABLE)
					.classed("view-disabled", !OPEN_WITH_TABLE);


				// initialize chart view
				initialize_chart_view(ORIGINAL_DATASET, CURRENT_VIS_TYPE);
				if (!OPEN_WITH_CHART) {
					hide_view("chart");
				}
				d3.select("#chart-btn")
					.classed("view-mngmt-btn-hidden", false)
					.classed("view-enabled", OPEN_WITH_CHART)
					.classed("view-disabled", !OPEN_WITH_CHART);

				// initialize image view
				initialize_image_view(ORIGINAL_CASE_LIST);
				if (!OPEN_WITH_IMAGE) {
					hide_view("image");
				}
				d3.select("#image-btn")
					.classed("view-mngmt-btn-hidden", false)
					.classed("view-enabled", OPEN_WITH_IMAGE)
					.classed("view-disabled", !OPEN_WITH_IMAGE);



				// initialize umap view
				// initialize_umap_view(ORIGINAL_DATASET);
				initializeUmapView(ORIGINAL_DATASET);
				if (!OPEN_WITH_UMAP) {
					hide_view("umap");
				}
				d3.select("#umap-btn")
					.classed("view-mngmt-btn-hidden", false)
					.classed("view-enabled", OPEN_WITH_UMAP)
					.classed("view-disabled", !OPEN_WITH_UMAP);


				$("#view-mngmt-btn-group").css("display", "block");
				d3.select("#page-title")
					.classed("mr-md-auto", false)
					.classed("mr-md-3", true);


				console.log("[LOG] App initialized.");
				APP_INITIALIZED = true;
			} else {
				console.log("waiting for image type checking ...");
			}
		}, 500);
	}
}

function data_sorting (keyword, desc=false) {
	var compare = function (a, b) {
		if (a[keyword] < b[keyword]) {
			if (desc) {
				return 1;
			} else {
				return -1;
			}
		} else if (a[keyword] > b[keyword]) {
			if (desc) {
				return -1;
			} else {
				return 1;
			}
		} else {
			return 0;
		}
	}

	CURRENT_SORT_ATTRIBUTE = keyword;
	ORIGINAL_DATASET.sort(compare);
	ORIGINAL_CASE_LIST = ORIGINAL_DATASET.map(function (d) {return d["Participant"];});
	CURRENT_MULTI_SELECTED.sort(compare);
	CURRENT_CASE_LIST = CURRENT_MULTI_SELECTED.map(function (d) {return d["Participant"];});
}



