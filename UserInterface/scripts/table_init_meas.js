function initialize_data_table_meas (dataset) {

	show_view("table_meas");
	var $table_meas = $("#result-table_meas");

	generate_table_meas(dataset, $table_meas);
	generate_config_meas(dataset);

	TABLEM = $table_meas.DataTable(DATA_TABLE_CONFIG_meas);

	// ORIGINAL_FEATURE_LIST.length = 11;

	init_visibility_meas();
	// init_editability_meas();
	init_button_style();

	CURRENT_SORT_ATTRIBUTE = ORIGINAL_FEATURE_LIST2[TABLEM.order()[0][0]];





	$table_meas.find("tbody").on("click", 'td', function () {

		if ($(TABLEM.column($(this).index() + ":visIdx").header()).text().trim() != "comments") {
			var case_name = $(this).parent().find("td:first-child").text();

			if (case_name != CURRENT_SELECTED) {
				enter_select_mode(case_name, true);
			} else {
				exit_select_mode();
			}
		} else {
			$("tr.selected").removeClass("selected");
		}
	});

	$(".dataTables_scrollHeadInner > table > thead > tr > th").on("click", function () {
		data_sorting($(this).text(), (TABLEM.order()[0][1] == 'desc'));
		update_views();
	});
}


function generate_table_meas (dataset, table) {
	
	var thead_content = "<tr>";

	ORIGINAL_FEATURE_LIST2.forEach(function (d, i) {
		thead_content += ("<th>" + d + "</th>");
	});
	thead_content += "</tr>";

	tbody_content = "";
	for (var i = 0; i < dataset.length; i++) {
		tbody_content += "<tr>";
		for (var j = 0; j < ORIGINAL_FEATURE_LIST2.length; j++) {
			tbody_content += ("<td>" + dataset[i][ORIGINAL_FEATURE_LIST2[j]] + "</td>");
		}
		tbody_content += "</tr>";
	}

	table.children("thead").empty().html(thead_content);
	table.children("tbody").empty().html(tbody_content);
}


function generate_config_meas (dataset) {

	// 1. named column
	// 2. customized colvis

	var colvis_action = function (e, dt, node, config) {
		var column_name = node[0].text;
		if (this.active()) {
			// update the table column
			this.active(false);
			TABLEM.column(column_name + ":name").visible(false);
			
			CURRENT_HIDDEN_COLUMNS.push(column_name);
			
			// update parallel coordinate -> delete from CURRENT_PARAC_ATTRIBUTES
			CURRENT_PARAC_ATTRIBUTES = generate_current_parac_attributes();
			update_chart_view("parallel_coordinate", CURRENT_MULTI_SELECTED);

		} else {
			// update the table column
			this.active(true);
			TABLEM.column(column_name + ":name").visible(true);
			
			var index = CURRENT_HIDDEN_COLUMNS.indexOf(column_name);
			if (index > -1) {
				CURRENT_HIDDEN_COLUMNS.splice(index, 1);
			} else {
				console.log("[DEBUG] " + column_name + " is not in CURRENT_HIDDEN_COLUMNS.")
			}

			// update parallel coordinate
			CURRENT_PARAC_ATTRIBUTES = generate_current_parac_attributes();
			update_chart_view("parallel_coordinate", CURRENT_MULTI_SELECTED);

		}
	};

	DATA_TABLE_CONFIG_meas["columns"] = [];
	var colvis_buttons_config = []; // customized colvis buttons list (every header) 

	ORIGINAL_FEATURE_LIST2.forEach(function (header) {
		DATA_TABLE_CONFIG_meas["columns"].push({
			name: header
		});
		colvis_buttons_config.push({
			text: header,
			className: DEFAULT_HIDDEN_COLUMNS.indexOf(header) == -1 ? 'active' : null,
			action: colvis_action
		});
	});

	var colvis_config = {
		extend: 'collection',
		text: 'Measures',
		buttons: colvis_buttons_config,
		fade: 500
	};

	DATA_TABLE_CONFIG_meas["buttons"].push(colvis_config);
}


function init_visibility_meas () {
	DEFAULT_HIDDEN_COLUMNS.forEach(function (hidden_header) {
		TABLEM.column(hidden_header + ":name").visible(false);
	});
}


function init_button_style () {
	$(".table-control > div.dt-buttons").removeClass("btn-group").addClass("btn-group-vertical");
	// $(".table-control > div.dt-buttons");
	$(".table-control > div.dt-buttons > button").removeClass("btn-secondary").addClass("btn-outline-secondary");
}


function select_row_in_table_meas (case_name, from_tablem) {
	if (from_tablem) return;

	var offset = 0;

	TABLEM.$("tr.selected").removeClass("selected");
	var target_index = TABLEM.row(function(idx, data, node) {
		if (data[0] == case_name) {
			return true;
		} else {
			return false;
		}
	}).select().index();

	TABLEM.row(target_index + offset).scrollTo();
}


function update_multi_selected_table_meas_view (case_names) {
	TABLEM.clear();
	TABLEM.rows.add(CURRENT_MULTI_SELECTED.map(function(d) {return Object.values(d);})).draw();
}
