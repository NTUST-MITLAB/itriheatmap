<?php
//$files = scandir("./");
//print_r($files);
$path = $_POST['getAjax'];
$dir = "./".$path;
$handler = opendir($dir);
    while (($filename = readdir($handler)) !== false) {//务必使用!==，防止目录下出现类似文件名“0”等情况
        if ($filename != "." && $filename != ".." && strpos($filename, ".") == false) {
                $files[] = $filename ;
           }
       }  
    closedir($handler);
     
//打印所有文件名
foreach ($files as $value) {
    echo $value." ";
}
//echo json_encode(array("result"=>$files));

?>