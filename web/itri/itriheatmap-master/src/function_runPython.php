<?php
//$files = scandir("./");
//print_r($files);
$python_command = $_POST['params'];
//$python_command = "C:\\xampp\\htdocs\\itri\\itriheatmap-master\\src\\web_pci.py 1 37 2>&1";
//$data = explode(",", $path);
//$data_string = "";
//foreach ($data as $value) {
//	$data_string .= ($value." ");
//}

$exec_string = "C:\Users\Ben\Anaconda3\python.exe " . $python_command;
$output = system($exec_string);
//echo $output;
     
//打印所有文件名
//foreach ($files as $value) {
//    echo $value." ";
//}
//echo json_encode(array("result"=>$files));

?>