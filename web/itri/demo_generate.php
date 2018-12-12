<!DOCTYPE html>
<!-- saved from url=(0061)https://bootstrap.hexschool.com/docs/4.1/examples/dashboard/# -->
<html lang="en"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">


    
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="icon" href="https://bootstrap.hexschool.com/favicon.ico">

    <title>zzz</title>

    <!-- Bootstrap core CSS -->
    <link href="./skr/bootstrap.min.css" rel="stylesheet">
    <!-- Time Series bar CSS -->
    <style>
    .time_series{
      width:700px;
      height:20px;
    }
    </style>
    
    <!-- Custom styles for this template -->
    <link href="./skr/dashboard.css" rel="stylesheet">
  <style type="text/css">/* Chart.js */
@-webkit-keyframes chartjs-render-animation{from{opacity:0.99}to{opacity:1}}@keyframes chartjs-render-animation{from{opacity:0.99}to{opacity:1}}.chartjs-render-monitor{-webkit-animation:chartjs-render-animation 0.001s;animation:chartjs-render-animation 0.001s;}</style></head>

  <body>
    <nav class="navbar navbar-dark fixed-top bg-dark flex-md-nowrap p-0 shadow">
      <a class="navbar-brand col-sm-3 col-md-2 mr-0" href="http://mit.et.ntust.edu.tw:8080">MITLAB</a>
      <h1 style="color:white;">Please select the option and click the button</h1>
      <ul class="navbar-nav px-3">
        <li class="nav-item text-nowrap">
          <a class="nav-link">Login</a>
        </li>
      </ul>
    </nav>

    <div class="container-fluid">
      <div class="row">
        <nav class="col-md-2 d-none d-md-block bg-light sidebar">
          <div class="sidebar-sticky">
            <ul class="nav flex-column">
              <li class="nav-item">
                <a class="nav-link active">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-home"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
                  Dashboard <span class="sr-only">(current)</span>
                </a>
              </li>
              <li class="nav-item" style="display: none">
                <a class="nav-link">                  
                  <select id='priority' onchange="addList(); invisibleMovie()" style="width: 150px; font-size: 24px">                  
                  <option value="priority_1">priority 1</option>
                  <option value="priority_2">priority 2</option>
                  <option value="priority_3">priority 3</option>
                  <option value="priority_4">priority 4</option>
                  <option value="priority_5">priority 5</option>   
                  <option value="priority_6">priority 6</option>   
                  </select>
                </a>
              </li>
              <li class="nav-item" style="display: none">
                <a class="nav-link">                                    
                  <select id='set' style="width: 150px; font-size: 24px" onchange="invisibleMovie()">
                  <option value="set1">set 1</option>
                  <option value="set2">set 2</option>
                  <option value="set3">set 3</option>
                  <option value="set4">set 4</option>
                  <option value="set5">set 5</option>   
                  <option value="set6">set 6</option> 
                  <option value="set7">set 7</option>
                  </select>
                </a>
              </li>                          
              <li class="nav-item" style="display: none;">
                <a class="nav-link">
                  <form>
                  <input type="radio" name="signal" value="Multi" checked onclick="showPCI(0); invisibleMovie()"> Multi PCI
                  <br>
                  <input type="radio" name="signal" value="Individual" onclick="showPCI(1); invisibleMovie()"> Individual PCI
                  </form>
                </a>
              </li>
              <div id="div1" style="display:none">
              <li class="nav-item">
                <a class="nav-link">
                  <select id='pci_number' style="width: 150px; font-size: 24px" onchange="invisibleMovie()">
                  <option value="301">301</option>
                  <option value="302">302</option>
                  </select>
                </a>
              </li>
              </div>         
              <li class="nav-item">
                <a class="nav-link">
                  <select id='jpg_kind' style="width: 150px; font-size: 24px" onchange="invisibleMovie()">
                  <option value="pci">PCI</option>
                  <option value="rsrp">RSRP</option>
                  <option value="rsrq">RSRQ</option>
                  <option value="snr">SNR</option>                  
                  </select>
                </a>
              </li>
              <li>
                <a class="nav-link">
                  <form name="registrationForm">
                    <p>AP1</p>
                     <input class="time" type="range" name="apInputName1" id="apInputId1" value="10" min="-5" max="20" oninput="apOutputId1.value = apInputId1.value">
                     <output name="apOutputName1" id="apOutputId1">10</output>
                    <p>AP2</p>
                     <input type="range" name="apInputName2" id="apInputId2" value="10" min="-5" max="20" oninput="apOutputId2.value = apInputId2.value">
                     <output name="apOutputName2" id="apOutputId2">10</output>
                    <p>AP3</p>
                     <input type="range" name="apInputName3" id="apInputId3" value="10" min="-5" max="20" oninput="apOutputId3.value = apInputId3.value">
                     <output name="apOutputName3" id="apOutputId3">10</output>
                    <p>AP4</p>
                     <input type="range" name="apInputName4" id="apInputId4" value="10" min="-5" max="20" oninput="apOutputId4.value = apInputId4.value">
                     <output name="apOutputName4" id="apOutputId4">10</output>
                    <p>AP5</p>
                     <input type="range" name="apInputName5" id="apInputId5" value="10" min="-5" max="20" oninput="apOutputId5.value = apInputId5.value">
                     <output name="apOutputName5" id="apOutputId5">10</output>
                    <p>AP6</p>
                     <input type="range" name="apInputName6" id="apInputId6" value="10" min="-5" max="20" oninput="apOutputId6.value = apInputId6.value">
                     <output name="apOutputName6" id="apOutputId6">10</output>
                     <br>
                     <br>
                  </form>
              </a> 
              </li>               
              <li class="nav-item">
                <a class="nav-link">
                  <button type="button" onclick="RunPythonAlgorithm()" style="font-size: 26px">Click me!</button>                      
                  <div id="div_btn_movie" style="display: none;">
                    <br>
                    <br>
                    <button type="button" onclick="generateMovie()" style="font-size: 26px; display: block;">Movie</button>                    
                    <div style="display: inline;">
                      <label for="movie">Time Interval (1 ~ 10s) : </label>
                      <input id="movie_time_interval" type="number" value="5" min="1" max="10" width="10" style="font-size: 16px" >s</input>
                    </div>                    
                  </div>
                </a>
                <br>
                <p id="run_time" style="border-style:solid; font-size: 20px">  Run Time :</p>                
              </li>
            </ul>
          </div>
        </nav>

        <main role="main" class="col-md-9 ml-sm-auto col-lg-10 px-4"><div class="chartjs-size-monitor" style="position: absolute; left: 0px; top: 0px; right: 0px; bottom: 0px; overflow: hidden; pointer-events: none; visibility: hidden; z-index: -1;"><div class="chartjs-size-monitor-expand" style="position:absolute;left:0;top:0;right:0;bottom:0;overflow:hidden;pointer-events:none;visibility:hidden;z-index:-1;"><div style="position:absolute;width:1000000px;height:1000000px;left:0;top:0"></div></div><div class="chartjs-size-monitor-shrink" style="position:absolute;left:0;top:0;right:0;bottom:0;overflow:hidden;pointer-events:none;visibility:hidden;z-index:-1;"><div style="position:absolute;width:200%;height:200%;left:0; top:0"></div></div></div>
          <div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
            <p class="h2">Heat map</p>
            <div class="btn-toolbar mb-2 mb-md-0">
              <div class="btn-group mr-2">
                <button class="btn btn-sm btn-outline-secondary">Share</button>
                <button class="btn btn-sm btn-outline-secondary">Export</button>
              </div>
            </div>
          </div>
          <h1 id="photo1_title" class="h2" style="display: inline;"></h1>
          <div id="div_btn_movie_d" style="display: none;">
            <a class="nav-link" style="display: inline;">              
              <button type="button" onclick="generateMovie()" style="font-size: 16px">Movie</button>
            </a>
          </div>
          <img id="photo1"; src="./itriheatmap-master/empty.jpg" onerror="this.src='./itriheatmap-master/empty.jpg'" style="display: block;">
          <h1 id="photo2_title" class="h2"></h1>          
          <img id="photo2"; src="" onerror="this.src='./itriheatmap-master/empty.jpg'">          
          <div id="div_slide1" style="display: none">
            <h1 id="slide1_photo_title" class="h2"></h1>
            <img id="slide1_photo" src="">
            <div id="div_slide1_bar">
              <br>
              <input class="time_series" type="range" name="time1_input" id="time1_input" value="10" min="0" max="100" step="1" oninput="showMovie()">
              <output name="time1_output" id="time1_output">50</output>
              <br>
              <input id="checkbox_movie_autoplay" type="checkbox" onchange="autoplayMovie()" style="font-size: 20px">Auto Play</input>              
              <input id="movie_speed" type="number" value="400" min="50" max="3000" style="font-size: 16px"> ms</input>
              <button onclick="add_movie_speed(100)">+100</button>
              <button onclick="add_movie_speed(-100)">-100</button>
              <br>
              <br>              
            </div>            
          </div>
          <div id="div_slide2" style="display: none">
            <img id="slide2_photo" src="">            
            <input class="time_series" type="range" name="time2_input" id="time2_input" value="10" min="0" max="100" step="5" oninput="time2_output.value = time2_input.value">
            <output name="time2_output" id="time2_output">50</output>  
          </div>          
                    
        </main>
      </div>
    </div>


    <!-- Put button function in here-->
    <script src="https://code.jquery.com/jquery-3.3.1.min.js" type="text/javascript"></script>
    <script type="text/javascript">
    function add_movie_speed(value){
      document.getElementById("movie_speed").stepUp(value);
      //
      var checked = document.getElementById("checkbox_movie_autoplay").checked;      
      if(checked==true){
        clearInterval(movie_autoplay_id);
        movie_autoplay_id = setInterval(autoplayMovieHandler, document.getElementById("movie_speed").value);
      }
    }

    $("").trigger("click");
    function addList()
    {               
    var pci_range = $("input[name=signal]:checked").val(); 
    var path = "";
    var priority = document.getElementById("priority").value;    
    switch(priority){
        case 'priority_1':
        case 'priority_2':
        case 'priority_3':
        case 'priority_4':
        case 'priority_5':
            path = 'itriheatmap-master/demo_'+priority+'/images/';   
            changePciNumber(1);
            changeJpgKind(1);
            break;
        case 'priority_6':            
            path = "itriheatmap-master/all floor(individual)/images/"
            changePciNumber(6);
            changeJpgKind(6);
            break;
    }   
    url = "./function_getfolder.php"     
    $.post(url,{"getAjax":path}, function(data) {        
        var setObj = document.getElementById("set");
        while(setObj.length>0)
            setObj.remove(0);
        data = data.replace(/set/g, '');        
        var array = data.split(' ');
        array.sort(function(a, b){return a-b});
        for (i = 1; i<array.length; i++)
        {            
            var opt = document.createElement("option");
            opt.text = 'set '+array[i];
            opt.value = "set"+array[i];
            setObj.options.add(opt);
        }
    });    
    }    

    function sleep(milliseconds) 
    { 
    var start = new Date().getTime(); 
    while(1)
         if ((new Date().getTime() - start) > milliseconds)
              break;
    }    

    function changePciNumber(index){
      var setObj = document.getElementById("pci_number");
      while(setObj.length>0)
            setObj.remove(0);
      if(index != 6)
        pci_number_array = [301, 302];      
      else
        pci_number_array = [37, 38, 39, 40, 41, 42];
      for(i = 0; i<pci_number_array.length; i++)
      {
        var opt = document.createElement("option");
        opt.text = pci_number_array[i];
        opt.value = pci_number_array[i];
        setObj.options.add(opt);
      }
    }

    function changeJpgKind(index){
      var setObj = document.getElementById("jpg_kind");
      if(index != 6)
        for (i=0; i<2; i++)
          setObj.options.remove(4);
      else
      { 
        if(setObj.options.length<6)
        {
          var opt = document.createElement("option");
          opt.text = "Interference";
          opt.value = "interference";
          setObj.options.add(opt);
          var opt = document.createElement("option");
          opt.text = "Mode";
          opt.value = "mode";
          setObj.options.add(opt);
        }
        
      }
    }

    function test(){
        //alert();
        a = "itriheatmap-master/demo_priority_1/images/";
        //alert(a);
        //getSets(a);
        document.getElementById('div1').style.display ='none';
    }
    function showPCI(select){
      if(select == 1)
      {
        document.getElementById('div1').style.display ='block';
        changeJpgKind(1);
      }
      else
      {
        document.getElementById('div1').style.display ='none';
        changeJpgKind(6);
      }
    }
    
    function invisibleMovie(){
      document.getElementById("div_btn_movie").style.display="none";
    }

    function generateMovie(){      
      //
      var priority = document.getElementById("priority").value; 
      var set = document.getElementById("set").value;
      var pci_range = $("input[name=signal]:checked").val();
      var pci_number = document.getElementById("pci_number").value;
      var jpg_kind = document.getElementById("jpg_kind").value;
      var time_interval = document.getElementById("movie_time_interval").value;
      //
      movie_set = set;
      movie_pci_range = pci_range;
      movie_pci_number = pci_number;
      movie_jpg_kind = jpg_kind;
      movie_time_interval = time_interval;          
      //
      if(priority=="priority_6" && movie_flag_generating==0){
        //
        movie_flag_generating = 1;
        //
        document.getElementById("div_slide1").style.display="block";
        document.getElementById("div_slide1_bar").style.display="none";
        document.getElementById("slide1_photo_title").innerHTML="";
        document.getElementById("slide1_photo").src="./itriheatmap-master/loading.gif";
        document.getElementById("run_time").innerHTML="Run Time : Please Waiting ...";
        python_path = " web_"+jpg_kind+"_over_time.py";      
        param1 = set.replace("set", "");
        param2 = pci_number;
        param3 = movie_time_interval;
        if (pci_range == "Multi")
          param2 = 0;
        python_params = " " + param1 + " " + param2 + " " +param3;
        python_command = python_path + python_params;
        url = "./itriheatmap-master/src/function_runPython.php";
        d1 = new Date().getTime();              
        $.post(url,{"params":python_command}, function(data) { 
            movie_d = new Date();
            setMoiveDir();
            d2 = new Date().getTime();
            d3 = new Date(d2-d1);                      
            var m = (d3.getMinutes()<10 ? '0':'') + d3.getMinutes();
            var s = (d3.getSeconds()<10 ? '0':'') + d3.getSeconds();
            document.getElementById("run_time").innerHTML="Run Time : " + m + ":" + s;
            //
            movie_flag_generating = 0;
          });        
      }          
    }   

    var movie_folder_path_init = "itriheatmap-master/results/demo_";
    var movie_image_path = "";
    var movie_interval = 5;
    var movie_set = "";
    var movie_pci_range = "";
    var movie_pci_number = "";
    var movie_jpg_kind = "";    
    var movie_autoplay_id = 0;
    var movie_d;
    var movie_flag_generating = 0;
    var movie_autoplay_m = 0;
    function setMoiveDir(){
      var priority = document.getElementById("priority").value; 
      var set = document.getElementById("set").value;
      var pci_range = $("input[name=signal]:checked").val();
      var pci_number = document.getElementById("pci_number").value;
      var jpg_kind = document.getElementById("jpg_kind").value;
      movie_folder_path = (movie_folder_path_init + priority + "/movie_element");
      if(pci_range=="Individual")
          movie_folder_path = (movie_folder_path + "/" + set);
      url = "./function_getfile.php";    
      $.post(url,{"getAjax":movie_folder_path}, function(data) {            
          var array = data.split(' ');
          var slide1 = document.getElementById("time1_input");
          var slide1_text = document.getElementById("time1_output");
          slide1.max = (array.length-2);          
          slide1.value=0;
          slide1_text.innerHTML=0+" s";  
          movie_autoplay_m = 0;        
          document.getElementById("div_slide1").style.display="block";  
          document.getElementById("slide1_photo_title").innerHTML="Slide Movie of "+jpg_kind.toUpperCase();        
          document.getElementById("slide1_photo").src=getInitMovieFile()+"?a="+movie_d.getTime();
          document.getElementById("div_slide1_bar").style.display="block";
          //alert(getInitMovieFile());
      });
    } 

    function showMovie(){    
      //依照slide bar version  
      var slide1 = document.getElementById("time1_input");
      var slide1_text = document.getElementById("time1_output");
      image_name = "";
      if(movie_pci_range=="Multi")
        image_name = movie_set+"_"+movie_jpg_kind+"_";
      else
        image_name = movie_pci_number+"_"+movie_jpg_kind+"_";      
      index = (slide1.value);
      image_path = movie_folder_path + "/" + image_name + index + ".png";
      slide1_text.innerHTML=(index*movie_time_interval)+" s";
      document.getElementById("slide1_photo").src=image_path+"?a="+movie_d.getTime(); 
    }

    function getInitMovieFile(){
      image_name = "";
      if(movie_pci_range=="Multi")
        image_name = movie_set+"_"+movie_jpg_kind+"_";
      else
        image_name = movie_pci_number+"_"+movie_jpg_kind+"_";       
      image_path = movie_folder_path + "/" + image_name + "0.png";
      return image_path;
    }
    
    function autoplayMovieHandler(){
      var time1_input = document.getElementById("time1_input");
      time1_input.value = movie_autoplay_m;
      showMovie();
      movie_autoplay_m+=1;      
      if(movie_autoplay_m>Number(time1_input.max))
        movie_autoplay_m=0;
    }

    function autoplayMovie(){
      var checked = document.getElementById("checkbox_movie_autoplay").checked;
      //alert(checked);
      if(checked==true)
        movie_autoplay_id = setInterval(autoplayMovieHandler, document.getElementById("movie_speed").value);
      else
        clearInterval(movie_autoplay_id);
    }

    function RunPythonAlgorithm(){
      var priority = "priority_6"; 
      var set = document.getElementById("set").value;
      var pci_range = $("input[name=signal]:checked").val();
      var pci_number = document.getElementById("pci_number").value;
      var jpg_kind = document.getElementById("jpg_kind").value;
      var param_p1 = document.getElementById("apInputId1").value;
      var param_p2 = document.getElementById("apInputId2").value;
      var param_p3 = document.getElementById("apInputId3").value;
      var param_p4 = document.getElementById("apInputId4").value;
      var param_p5 = document.getElementById("apInputId5").value;
      var param_p6 = document.getElementById("apInputId6").value;

      python_path = "./";
      python_filename = "web_regressor-heatmap-generator";
      param1=1;
      param2=0;
      param3=0;
      switch(jpg_kind)
      {
        case "pci":
          param1="";
          python_filename = "web_classifier-heatmap-generator";
          break;
        case "rsrp":
          param1=1;
          break;
        case "rsrq":
          param1=2;
          break;
        case "snr":
          param1=3;
          break;
      }

      python_path = python_filename + ".py";          
      python_params = " " + param1 + " " + param2 + " " + param3 + " " + param_p1 + " " + param_p2 + " " + param_p3 + " " + param_p4 + " " + param_p5 + " " + param_p6;
      /*
      param1 -> 1:RSRP, 2:RSRQ, 3:SNR
      param2 -> 0:lgbm, 1:xgboost
      param3 -> 0:baseline, 1:independent_set_%d, 2:transfer_except_%d
      */

      //alert(python_path + " "+ python_params);

      document.getElementById("movie_time_interval").value = 5;
      document.getElementById("photo1_title").innerHTML="";
      document.getElementById("photo2_title").innerHTML="";
      document.getElementById("run_time").innerHTML="Run Time : Please Waiting ...";
      document.getElementById("div_btn_movie").style.display="none";
      document.getElementById("div_slide1").style.display="none";
      document.getElementById("div_slide2").style.display="none";
      document.getElementById("photo1").src="./itriheatmap-master/loading.gif";
      document.getElementById("photo2").src="./itriheatmap-master/empty.jpg";
      document.getElementById("checkbox_movie_autoplay").checked=false;
      document.getElementById("movie_speed").value=400;
      clearInterval(movie_autoplay_id);

      //alert(python_path);
      python_command = python_path + python_params;
      url = "./itriheatmap-master/src/function_runPython.php";    
      //url = "";
      //alert(python_command);
      d1 = new Date().getTime();
      if(priority == "priority_6"){
        //alert(python_command);                
        $.post(url,{"params":python_command}, function(data) { 
          var d = new Date();
          findJPG(d);
          d2 = new Date().getTime();
          d3 = new Date(d2-d1);
          var m = (d3.getMinutes()<10 ? '0':'') + d3.getMinutes();
          var s = (d3.getSeconds()<10 ? '0':'') + d3.getSeconds();
          document.getElementById("run_time").innerHTML="Run Time : " + m + ":" + s;                          
          });
      }
          
    }

    function findJPG(d){
      var priority = document.getElementById("priority").value; 
      var set = document.getElementById("set").value;
      var pci_range = $("input[name=signal]:checked").val();
      var pci_number = document.getElementById("pci_number").value;
      var jpg_kind = document.getElementById("jpg_kind").value;
      path1 = "";
      path2 = "";

      path = "itriheatmap-master/results/predicted/priority_6_set_%d.png";
      path = "itriheatmap-master/results/predicted/result.png";
      //path1 = path+filename+"_mean.png";            
      //path2 = path+filename+"_std.png";
      path1 = path;
      document.getElementById("photo1_title").innerHTML=jpg_kind.toUpperCase();
      //document.getElementById("photo2_title").innerHTML="Standard deviation of " + jpg_kind.toUpperCase();
      document.getElementById("photo1").src=path1+"?a="+d.getTime();
      //document.getElementById("photo2").src=path2+"?a="+d.getTime();         
    }

    //Folder is "results"
    function findJPG_v_results(d){
      var priority = document.getElementById("priority").value; 
      var set = document.getElementById("set").value;
      var pci_range = $("input[name=signal]:checked").val();
      var pci_number = document.getElementById("pci_number").value;
      var jpg_kind = document.getElementById("jpg_kind").value;
      path1 = "";
      path2 = "";
      //if(priority == "priority_6")
      //  RunPythonAlgorithm(priority, set, pci_range, pci_number, jpg_kind);
      //      
      if(priority=="priority_6" && jpg_kind!="interference" && jpg_kind!="mode")
        document.getElementById('div_btn_movie').style.display ='inline';      
      else
        document.getElementById('div_btn_movie').style.display ='none';      
      //
      if(pci_range != "Multi")
      {        
          if(priority != "priority_6")
            path = "itriheatmap-master/demo_"+priority+"/images/"+set+"/";  
          else
            path = "itriheatmap-master/results/demo_"+priority+"/images/"+set+"/";
          filename = pci_number+"_"+jpg_kind;     
          path1 = path+filename;
          if(jpg_kind != "pci")
          {
            path1 = path+filename+"_mean.png";            
            path2 = path+filename+"_std.png";
            document.getElementById("photo1_title").innerHTML="Mean of " + jpg_kind.toUpperCase();
            document.getElementById("photo2_title").innerHTML="Standard deviation of " + jpg_kind.toUpperCase();            
            document.getElementById('div_slide1').style.display ='none';        
            document.getElementById('div_slide2').style.display ='none';             
          }
          else
          {
            path1 = path+filename+".png";
            document.getElementById("photo1_title").innerHTML="PCI of Locations";
            document.getElementById("photo2_title").innerHTML="";                            
          }
          //alert(path1);
      }
      else
      {
        if(priority != "priority_6")
          path = "itriheatmap-master/demo_"+priority+"/images/";            
        else
          path = "itriheatmap-master/results/demo_"+priority+"/images/";
        filename = set+"_"+jpg_kind;
        path1 = path+filename;
        if(jpg_kind != "pci")
        {
        if(jpg_kind == "interference")
          {
            filename = set+"_pci_"+jpg_kind;
            path1 = path+filename+"_level.png";            
            path2 = path+filename+"_ratio.png";
            document.getElementById("photo1_title").innerHTML="Level of " + jpg_kind.toUpperCase();
            document.getElementById("photo2_title").innerHTML="Ratio of " + jpg_kind.toUpperCase();
          }
          else if(jpg_kind == "mode")
          {
            filename = set+"_pci_"+jpg_kind;
            path1 = path+filename+".png";
            document.getElementById("photo1_title").innerHTML=jpg_kind.toUpperCase()+" of PCI";
            document.getElementById("photo2_title").innerHTML="";
          }
          else
          {
            path1 = path+filename+"_mean.png";
            path2 = path+filename+"_std.png";
            document.getElementById("photo1_title").innerHTML="Mean of " + jpg_kind.toUpperCase();
            document.getElementById("photo2_title").innerHTML="Standard deviation of " + jpg_kind.toUpperCase();
          }          
          document.getElementById('div_slide1').style.display ='none'; 
          document.getElementById('div_slide2').style.display ='none'; 
        }
        else
        {
          path1 = path+filename+".png";
          document.getElementById("photo1_title").innerHTML="PCI of Locations";
          document.getElementById("photo2_title").innerHTML="";                            
        }
      }      
      document.getElementById("photo1").src=path1+"?a="+d.getTime();
      document.getElementById("photo2").src=path2+"?a="+d.getTime();
      //alert(path1);
    }
    
    </script>
    <!-- Bootstrap core JavaScript
    ================================================== -->
    <!-- Placed at the end of the document so the pages load faster -->
    <script integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script>window.jQuery || document.write('<script src="../../../../assets/js/vendor/jquery-slim.min.js"><\/script>')</script>
    <script src="./skr/popper.min.js"></script>
    <script src="./skr/bootstrap.min.js"></script>

    <!-- Icons -->
    <script src="./skr/feather.min.js"></script>
    <script>
      feather.replace()
    </script>

</body></html>