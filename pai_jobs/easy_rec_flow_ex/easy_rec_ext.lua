function split(str, delimiter)
    if str==nil or str=='' or delimiter==nil then
        return nil
    end
    local result = {}
    for match in (str..delimiter):gmatch("(.-)"..delimiter) do
        table.insert(result, match)
    end
    return result
end

function join(list, delimiter)
  return table.concat(list, delimiter)
end

function match_str_in_list(list, str_pattern)
  for idx=1,#(list) do
    if string.find(list[idx], str_pattern) ~= nil then
      return idx
    end
  end
  return nil
end

function CheckOssValid(host, bucket)
    if host == nil or string.len(host) == 0 or
        bucket == nil or string.len(bucket) == 0 then
        return false
    end
    return true
end

function ParseOssUri(oss_uri, default_host)
    if string.len(oss_uri) > 6 and string.find(oss_uri, "oss://") == 1 then
        _,_,_path,file = string.find(oss_uri,"oss://(.*/)(.*)")
        if _path == nil or string.len(_path) == 0 then
            error("invalid oss uri: "..oss_uri..", should end with '/'")
        end
        _,_,bucket_host,dir = string.find(_path, "(.-)(/.*)")
        if (string.find(bucket_host, "%.")) then
            _,_,bucket,host = string.find(bucket_host, "(.-)%.(.*)")
        else
            bucket = bucket_host
            host = default_host
        end
        if not CheckOssValid(host, bucket) then
            error("invalid oss uri: "..oss_uri..", oss host or bucket not found")
        end
        root_dir = bucket..dir
        return host, root_dir, file
    end
    error("invalid oss uri: "..oss_uri)
end

function getEntry(script_in, entryFile_in, config, cluster, res_project, version)
  if string.len(entryFile_in) == 0 then
    error('entryFile is not set')
  end
  if script_in ~= nil and string.len(script_in) > 0 then
    script = script_in
    entryFile = entryFile_in
  else
    script = "odps://" .. res_project .. "/resources/easy_rec_ext_" .. version .. "_res.tar.gz"
    entryFile = entryFile_in
  end

  return script, entryFile
end

function checkConfig(config)
  if config == nil or config == '' then
    error('config must be set')
  end

  s1, e1 = string.find(config, 'oss://')
  s2, e2 = string.find(config, 'http')
  if s1 == nil and s2 == nil then
    error("config path should be url or oss path")
  end
end

function checkTable(table)
  s1, e1 = string.find(table, "/tables/")
  s2, e2 = string.find(table, "odps://")
  if s1 == nil or s2 == nil then
    error(string.format("invalid odps table path: %s", table))
  end
end

function checkOss(path)
  s1, e1 = string.find(path, "oss://")
  if s1 == nil then
    error(string.format("invalid oss path: %s", path))
  end
end

function check_run_mode(cluster, gpuRequired)
  if (cluster ~=nil and cluster ~= "") and gpuRequired ~=""  then
    error(string.format('cluster and gpuRequired should not be set at the same time. cluster: %s gpuRequired:%s',
          cluster, gpuRequired))
  end
end

function getHyperParams(config, cmd, checkpoint_path,
                        eval_result_path, export_dir,  gpuRequired,
                        cpuRequired, memRequired, cluster, continue_train,
                        distribute_strategy, with_evaluator, eval_method,
                        edit_config_json, selected_cols,
                        model_dir, hpo_param_path, hpo_metric_save_path,
                        saved_model_dir, all_cols, all_col_types,
                        reserved_cols, output_cols, model_outputs,
                        input_table, output_table, tables, query_table,
                        doc_table, knn_distance, knn_num_neighbours,
                        knn_feature_dims, knn_index_type, knn_feature_delimiter,
                        knn_nlist, knn_nprobe, knn_compress_dim, train_tables,
                        eval_tables, boundary_table, batch_size, profiling_file,
                        mask_feature_name, extra_params)
  hyperParameters = ""
  if cmd == "predict" then
    if cluster == nil or cluster == '' then
      error('cluster must be set')
    end
    if saved_model_dir == nil or saved_model_dir == '' then
      error('saved_model_dir must be set')
      checkOss(saved_model_dir)
    end
    hyperParameters = " --cmd=" .. cmd
    hyperParameters = hyperParameters .. " --saved_model_dir=" .. saved_model_dir
    hyperParameters = hyperParameters .. " --all_cols=" .. all_cols ..
                     " --all_col_types=" .. all_col_types
    if selected_cols ~= nil and selected_cols ~= '' then
      hyperParameters = hyperParameters .. " --selected_cols=" .. selected_cols
    end
    if reserved_cols ~= nil and string.len(reserved_cols) > 0 then
      hyperParameters = hyperParameters .. " --reserved_cols=" .. reserved_cols
    end
    hyperParameters = hyperParameters .. " --batch_size=" .. batch_size
    if profiling_file ~= nil and profiling_file ~= '' then
      checkOss(profiling_file)
      hyperParameters = hyperParameters .. " --profiling_file=" .. profiling_file
    end
    --support both 'probs float, embedding string' and 'probs, embedding' format
    --in easy_rec.python.inferece.predictor.predict_table
    if model_outputs ~= nil and model_outputs ~= "" then
      hyperParameters = hyperParameters .. " --output_cols='" .. model_outputs .. "'"
    else
      hyperParameters = hyperParameters .. " --output_cols='" .. output_cols .. "'"
    end
    checkTable(input_table)
    checkTable(output_table)

    if extra_params ~= nil and extra_params ~= '' then
      hyperParameters = hyperParameters .. extra_params
    end
    return hyperParameters, cluster, tables, output_table
  end

  if cmd == "vector_retrieve" then
    if cluster == nil or cluster == '' then
      error('cluster must be set')
    end
    checkTable(query_table)
    checkTable(doc_table)
    checkTable(output_table)
    hyperParameters = " --cmd=" .. cmd
    hyperParameters = hyperParameters .. " --batch_size=" .. batch_size
    hyperParameters = hyperParameters .. " --knn_distance=" .. knn_distance
    if knn_num_neighbours ~= nil and knn_num_neighbours ~= '' then
      hyperParameters = hyperParameters .. ' --knn_num_neighbours=' .. knn_num_neighbours
    end
    if knn_feature_dims ~= nil and knn_feature_dims ~= '' then
      hyperParameters = hyperParameters .. ' --knn_feature_dims=' .. knn_feature_dims
    end
    hyperParameters = hyperParameters .. " --knn_index_type=" .. knn_index_type
    hyperParameters = hyperParameters .. " --knn_feature_delimiter=" .. knn_feature_delimiter
    if knn_nlist ~= nil and knn_nlist ~= '' then
      hyperParameters = hyperParameters .. ' --knn_nlist=' .. knn_nlist
    end
    if knn_nprobe ~= nil and knn_nprobe ~= '' then
      hyperParameters = hyperParameters .. ' --knn_nprobe=' .. knn_nprobe
    end
    if knn_compress_dim ~= nil and knn_compress_dim ~= '' then
      hyperParameters = hyperParameters .. ' --knn_compress_dim=' .. knn_compress_dim
    end
    if extra_params ~= nil and extra_params ~= '' then
      hyperParameters = hyperParameters .. extra_params
    end
    return hyperParameters, cluster, tables, output_table
  end

  if cmd ~= "custom" then
    checkConfig(config)
  end

  hyperParameters = "--config='" .. config .. "'"

  if selected_cols ~= nil and selected_cols ~= '' then
    hyperParameters = hyperParameters .. ' --selected_cols=' .. selected_cols
  end

  hyperParameters = string.format('%s --cmd=%s', hyperParameters, cmd)

  if cmd == 'evaluate' then
    hyperParameters = hyperParameters .. " --checkpoint_path=" .. checkpoint_path
    hyperParameters = hyperParameters .. " --all_cols=" .. all_cols ..
                     " --all_col_types=" .. all_col_types
    hyperParameters = hyperParameters .. " --eval_result_path=" .. eval_result_path
    hyperParameters = hyperParameters .. " --mask_feature_name=" .. mask_feature_name
    hyperParameters = hyperParameters .. " --distribute_strategy=" .. distribute_strategy
  elseif cmd == 'export' then
    hyperParameters = hyperParameters .. " --checkpoint_path=" .. checkpoint_path
    hyperParameters = hyperParameters .. " --export_dir=" .. export_dir
  elseif cmd == 'train' then
    hyperParameters = hyperParameters .. " --all_cols=" .. all_cols ..
                     " --all_col_types=" .. all_col_types
    hyperParameters = hyperParameters .. " --continue_train=" .. continue_train
    hyperParameters = hyperParameters .. " --distribute_strategy=" .. distribute_strategy
    if with_evaluator ~= "" and tonumber(with_evaluator) ~= 0 then
      hyperParameters = hyperParameters .. " --with_evaluator"
    end
    if eval_method ~= 'none' and eval_method ~= 'separate' and eval_method ~= 'master' then
      error('invalid eval_method ' .. eval_method)
    end
    if eval_method ~= "" then
      hyperParameters = hyperParameters .. " --eval_method=" .. eval_method
    end

    -- tables used for train and evaluate
    if train_tables ~= "" and train_tables ~= nil then
      hyperParameters = hyperParameters .. " --train_tables " .. train_tables
    end
    if eval_tables ~= "" and eval_tables ~= nil then
      hyperParameters = hyperParameters .. " --eval_tables " .. eval_tables
    end
    if boundary_table ~= "" and boundary_table ~= nil then
      hyperParameters = hyperParameters .. " --boundary_table " .. boundary_table
    end

    if hpo_param_path ~= "" and hpo_param_path ~= nil then
      hyperParameters = hyperParameters .. " --hpo_param_path=" .. hpo_param_path
      if hpo_metric_save_path == nil then
        error('hpo_metric_save_path must be set')
      end
      hyperParameters = hyperParameters .. " --hpo_metric_save_path=" .. hpo_metric_save_path
    end
  end

  if edit_config_json ~= "" and edit_config_json ~= nil then
    hyperParameters = hyperParameters ..
        string.format(" --edit_config_json='%s'", edit_config_json)
  end

  if model_dir ~= "" and model_dir ~= nil then
    checkOss(model_dir)
    hyperParameters = hyperParameters .. " --model_dir=" .. model_dir
  end

  check_run_mode(cluster, gpuRequired)
  if gpuRequired ~= "" then
    num_gpus_per_worker = math.max(math.ceil(tonumber(gpuRequired)/100), 0)
    cluster = string.format('{"worker":{"count":1, "gpu":%s, "cpu":%s, "memory":%s}}',
                     gpuRequired, cpuRequired, memRequired)
  elseif cluster ~= "" then
    gpus_str = string.match(cluster, '"gpu"%s*:%s*(%d+)')
    if gpus_str ~= nil then
      num_gpus_per_worker = math.max(math.ceil(tonumber(gpus_str)/100), 0)
    else
      num_gpus_per_worker = 1
    end
  else
    num_gpus_per_worker = 1
  end
  hyperParameters = string.format("%s --num_gpus_per_worker=%s ", hyperParameters,
                                  num_gpus_per_worker)

  if extra_params ~= nil and extra_params ~= '' then
    hyperParameters = hyperParameters .. extra_params
  end

  return hyperParameters, cluster, tables, output_table
end

function splitTableParam(table_path)
  --  odps://xx_project/tables/table_name/pa=1/pb=2
  --  split table name and partitions
  delimiter = '/'
  eles = split(table_path, delimiter)
  project_name = eles[3]
  table_name = eles[5]
  local partitions = {}
  for i=6, table.getn(eles) do
    table.insert(partitions, eles[i])
  end
  partition_str = join(partitions, delimiter)

  return project_name, table_name, partition_str
end

function getInputTableColTypes(inputTable)
  -- to test: uncomment the following, and comment the rest
  --return {["a"] = "string", ["b"] = "int",["c"] = "string",["d"] = "int"}, {"a", "b", "c" }
  local all_input_cols  = Builtin.GetAllColumnNames(inputTable, ",")
  local all_input_types = Builtin.GetColumnDataTypes(inputTable, ",")
  local col_list = split(all_input_cols, ',')
  local type_list = split(all_input_types, ',')
  local col_map = {}
  for i=1,table.getn(col_list) do
    col_map[col_list[i]] = type_list[i]
  end
  return col_map, col_list
end

function getOutputCols(col_type_map, reserved_columns, result_column)
  local res_cols = split(reserved_columns, ',')
  local sql = "("
  if res_cols ~= nil then
    for i=1, table.getn(res_cols) do
      if col_type_map[res_cols[i]] == nil then
        error(string.format("column %s is not in input table", res_cols[i]))
        return
      else
        sql = sql .. res_cols[i] .. " " .. col_type_map[res_cols[i]] .. ","
      end
    end
  end
  sql = sql .. result_column .. " string)"
  return sql
end

function parseParitionSpec(partitions)
  local parition_names = {}
  local partition_values = {}
  local parts = split(partitions, "/")
  for i = 1, table.getn(parts) do
    local spec = split(parts[i], "=")
    if table.getn(spec) ~=2 then
      error("Partition Spec is not Right "..parts[i])
    else
      table.insert(parition_names, i, spec[1])
      table.insert(partition_values,i, spec[2])
    end
  end
  return parition_names, partition_values
end

function genCreatePartitionStr(partition_names)
  local part_str = "("
  for i = 1,#(partition_names) do
    part_str = part_str..partition_names[i].." string,"
  end
  part_str = string.sub(part_str, 1, -2)
  return part_str..")"
end

function genAddPartitionStr(parition_names, partition_values)
  local part_str = "("
  for i = 1, #(parition_names) do
    part_str= part_str..parition_names[i].."=\""..partition_values[i].."\","
  end
  part_str = string.sub(part_str, 1, -2)
  return part_str..")"
end


function parseTable(cmd, inputTable, outputTable, selectedCols, excludedCols,
                     reservedCols, lifecycle, outputCol, tables,
                     trainTables, evalTables, boundaryTable, queryTable, docTable)
  -- all_cols, all_col_types, selected_cols, reserved_cols,
  -- create_table_sql, add_partition_sql, tables parameter to runTF
  if cmd ~= 'train' and cmd ~= 'evaluate' and cmd ~= 'predict' and cmd ~= 'export'
     and cmd ~= 'evaluate' and cmd ~= 'custom' and cmd ~= 'vector_retrieve' then
    error('invalid cmd: ' .. cmd .. ', should be one of train, evaluate, predict, evaluate, export, custom, vector_retrieve')
  end

  -- for export
  if cmd == 'export' or cmd == 'custom' then
    return "", "", "", "", "select 1;", "select 1;", tables
  end

  -- merge all tables into all_tables
  all_tables  = {}
  table_id = 0
  if tables ~= nil and tables ~= ''
  then
    tmpTables = split(tables, ',')
    for k=1, table.getn(tmpTables) do
      v = tmpTables[k]
      if all_tables[v] == nil then
        all_tables[v] = table_id
        table_id = table_id + 1
      end
    end
    if inputTable == nil or inputTable == ''
    then
      inputTable = tmpTables[1]
    end
  end

  if cmd == 'vector_retrieve' then
    inputTable = queryTable
    all_tables[queryTable] = table_id
    table_id = table_id + 1
    all_tables[docTable] = table_id
    table_id = table_id + 1
  end

  if cmd == 'train' then
    -- merge train table and eval table into all_tables
    if trainTables ~= '' and trainTables ~= nil then
      tmpTables = split(trainTables, ',')
      for k=1, table.getn(tmpTables) do
        v = tmpTables[k]
        if all_tables[v] == nil then
          all_tables[v] = table_id
          table_id = table_id + 1
        end
      end
      inputTable = tmpTables[1]

      tmpTables = split(evalTables, ',')
      for k=1, table.getn(tmpTables) do
        v = tmpTables[k]
        if all_tables[v] == nil then
          all_tables[v] = table_id
          table_id = table_id + 1
        end
      end
    end
    if boundaryTable ~= nil and boundaryTable ~= '' then
      if all_tables[boundaryTable] == nil then
        all_tables[boundaryTable] = table_id
        table_id = table_id + 1
      end
    end
  end

  if cmd == 'evaluate' then
    -- merge evalTables into tables if evalTables is set
    if evalTables ~= nil and evalTables ~= ''
    then
      tmpTables = split(evalTables, ',')
      for k=1, table.getn(tmpTables) do
        v = tmpTables[k]
        if all_tables[v] == nil then
          all_tables[v] = table_id
          table_id = table_id + 1
        end
      end
      inputTable = tmpTables[1]
    end
  end

  if cmd == 'predict' then
    -- merge inputTable into all_tables if inputTable is set
    if inputTable ~= nil and inputTable ~= ''
    then
      tmpTables = split(inputTable, ',')
      for k=1, table.getn(tmpTables) do
        v = tmpTables[k]
        if all_tables[v] == nil then
          all_tables[v] = table_id
          table_id = table_id + 1
        end
      end
    else
      -- if inputTable is not set but tables is set
      -- set inputTable to tables
      if tables ~= '' and tables ~= nil then
        inputTable = split(tables, ',')[1]
      else
        error('either inputTable or tables must be set')
      end
    end
  end

  -- merge all_tables into tables
  tables = {}
  for k,v in pairs(all_tables) do
    -- ensure order to be compatible
    tables[v+1] = k
    --table.insert(tables, k)
  end

  if inputTable == nil or inputTable == '' then
    error('inputTable is not defined')
  end

  tables = join(tables, ',')

  if cmd == 'vector_retrieve' then
    if outputTable == nil or outputTable == '' then
      error("outputTable is not set")
    end

    proj1, table1, partition1 = splitTableParam(outputTable)
    out_table_name = proj1 .. "." .. table1
    create_sql = ''
    add_partition_sql = ''
    if partition1 ~= nil and string.len(partition1) ~= 0 then
      local partition_names, parition_values = parseParitionSpec(partition1)
      create_partition_str = genCreatePartitionStr(partition_names)
      create_sql = string.format("create table if not exists %s (query BIGINT, doc BIGINT, distance DOUBLE) partitioned by %s lifecycle %s;", out_table_name, create_partition_str, lifecycle)
      add_partition_sql = genAddPartitionStr(partition_names, parition_values)
      add_partition_sql = string.format("alter table %s add if not exists partition %s;", out_table_name, add_partition_sql)
    else
      create_sql = string.format("create table %s (query BIGINT, doc BIGINT, distance DOUBLE) lifecycle %s;", out_table_name, lifecycle)
      add_partition_sql = string.format("desc %s;",  out_table_name)
    end

    return "", "", "", "", create_sql, add_partition_sql, tables
  end

  -- analyze selected_cols excluded_cols for train, evaluate and predict
  proj0, table0, partition0 = splitTableParam(inputTable)
  input_col_types, input_cols = getInputTableColTypes(proj0 .. "." .. table0)

  if (excludedCols ~= nil and excludedCols ~= '') and
     (selectedCols ~= nil and selectedCols ~= '') then
    error('selected_cols and excluded_cols should not be set')
  end

  ex_cols_map = {}
  if excludedCols ~= '' and excludedCols ~= nil then
    ex_cols_lst = split(excludedCols, ',')
    for i=1, table.getn(ex_cols_lst) do
      ex_cols_map[ex_cols_lst[i]] = 1
    end
  end

  -- columns to be selected to input to the model
  selected_cols = {}
  -- all columns to read by TableRecordDataset
  all_cols = {}
  all_col_types = {}
  all_cols_map = {}
  if selectedCols ~= '' and selectedCols ~= nil then
    tmp_cols = split(selectedCols, ",")
  else
    tmp_cols = input_cols
  end

  for i=1, table.getn(tmp_cols) do
    if input_col_types[tmp_cols[i]] == nil then
      error(string.format("column %s is not in input table", tmp_cols[i]))
      return
    elseif ex_cols_map[tmp_cols[i]] == nil then
      -- not in excluded cols map
      if input_col_types[tmp_cols[i]] ~= nil and all_cols_map[tmp_cols[i]] == nil then
        table.insert(all_cols, tmp_cols[i])
        table.insert(all_col_types, input_col_types[tmp_cols[i]])
        table.insert(selected_cols, tmp_cols[i])
        all_cols_map[tmp_cols[i]] = 1
      end
    end
  end

  if cmd == 'train' or cmd == 'evaluate' then
    return join(all_cols, ","), join(all_col_types, ","),
         join(selected_cols, ","), '',
         "select 1;", "select 1;", tables
  end

  -- analyze reserved_cols for predict
  -- columns to be copied to output_table, may not be in selected columns
  -- could have overlapped columns with selected_cols and excluded_cols
  reserved_cols = {}
  reserved_col_types = {}
  if reservedCols ~= nil and reservedCols ~= '' then
    if reservedCols == 'ALL_COLUMNS' then
      tmp_cols = input_cols
    else
      tmp_cols = split(reservedCols, ',')
    end
    for i=1, table.getn(tmp_cols) do
      if input_col_types[tmp_cols[i]] ~= nil then
        table.insert(reserved_cols, tmp_cols[i])
        table.insert(reserved_col_types, input_col_types[tmp_cols[i]])
        if all_cols_map[tmp_cols[i]] == nil then
          table.insert(all_cols, tmp_cols[i])
          table.insert(all_col_types, input_col_types[tmp_cols[i]])
          all_cols_map[tmp_cols[i]] = 1
        end
      else
        error("invalid reserved_col: " .. tmp_cols[i] .. " available: " .. join(input_cols, ","))
      end
    end
  else
    table.insert(reserved_cols, selected_cols[0])
  end

  -- build create output table sql and add partition sql for predict
  sql_col_desc = {}
  for i=1, table.getn(reserved_cols) do
    table.insert(sql_col_desc, reserved_cols[i] .. " " .. reserved_col_types[i])
  end
  table.insert(sql_col_desc, outputCol)
  sql_col_desc = join(sql_col_desc, ",")

  if outputTable == nil or outputTable == '' then
    error("outputTable is not set")
  end

  proj1, table1, partition1 = splitTableParam(outputTable)
  out_table_name = proj1 .. "." .. table1
  create_sql = ''
  add_partition_sql = ''
  if partition1 ~= nil and string.len(partition1) ~= 0 then
    local partition_names, parition_values = parseParitionSpec(partition1)
    create_partition_str = genCreatePartitionStr(partition_names)
    create_sql = string.format("create table if not exists %s (%s) partitioned by %s lifecycle %s;", out_table_name, sql_col_desc, create_partition_str, lifecycle)
    add_partition_sql = genAddPartitionStr(partition_names, parition_values)
    add_partition_sql = string.format("alter table %s add if not exists partition %s;", out_table_name, add_partition_sql)
  else
    create_sql = string.format("create table %s (%s) lifecycle %s;", out_table_name, sql_col_desc, lifecycle)
    add_partition_sql = string.format("desc %s;",  out_table_name)
  end

  return join(all_cols, ","), join(all_col_types, ","),
         join(selected_cols, ","), join(reserved_cols, ","),
         create_sql, add_partition_sql, tables
end

function test_create_table()
  input_table = "odps://pai_rec_dev/tables/test_longonehot_4deepfm_20/part=1"
  output_table = "odps://pai_rec_dev/tables/test_longonehot_4deepfm_20_out/part=1"
  selectedCols = "a,b,c"
  excludedCols = ""
  reservedCols = "a"
  lifecycle=1
  outputCol = "score double"
  all_cols, all_cols_types, selected_cols, reserved_cols, create_sql, add_partition_sql =  createTable(input_table, output_table, selectedCols, excludedCols, reservedCols, lifecycle, outputCol)
  print(create_sql)
  print(add_partition_sql)
  print(string.format('all_cols = %s', all_cols))
  print(string.format('selected_cols = %s', selected_cols))
  print(string.format('reserved_cols = %s', reserved_cols))
end
--test_create_table()
