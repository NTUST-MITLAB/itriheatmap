<?php
//$files = scandir("./");
//print_r($files);
$path = $_POST['getAjax'];
$dir = "./".$path;
$files = scandir($dir, 1);
     
//打印所有文件名
foreach ($files as $value) {
	if(strpos($value, '.png') !== false)
    	echo $value." ";
}
//echo json_encode(array("result"=>$files));

?>