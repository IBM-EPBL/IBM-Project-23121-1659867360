<?php

@include 'config.php';

session_start();

if(isset($_POST['submit'])){

   $email = mysqli_real_escape_string($conn, $_POST['email']);
   $pass = md5($_POST['password']);
   $select = " SELECT * FROM user_form WHERE email = '$email' && password = '$pass' ";

   $result = mysqli_query($conn, $select);

   if(mysqli_num_rows($result) > 0){

      $row = mysqli_fetch_array($result);

      if($data['password'] == $password){
         header('location:http://127.0.0.1:5000/');
      }
     
   }else{
      $error[] = 'incorrect email or password!';
   }

};
?>

<!DOCTYPE html>
<html lang="en">
<head>
   <meta charset="UTF-8">
   <meta http-equiv="X-UA-Compatible" content="IE=edge">
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
   <title>login form</title>

   <!-- custom css file link  -->
   <link rel="stylesheet" href="register and login.css">
   <style>
      body{
      background: url(background.jpg); 
    background-size:cover;
    background-repeat: no-repeat;
      }
      </style>
</head>
<body>
   
<div class="form-container">

   <form action="" method="post">
   <div class="login-box">
   <img src="avatar.png" class="avatar">
   <h1>LOGIN</h1>
      <?php
      if(isset($error)){
         foreach($error as $error){
            echo '<span class="error-msg">'.$error.'</span>';
         };
      };
      ?>
      <p class="mail">E-Mail</p>
      <input type="email" name="email" required placeholder="Enter your email">
      <p class="password">Password</p>
      <input type="password" name="password" required placeholder="Enter your password">
      <input type="submit" name="submit" value="Login now" class="form-btn">
      <p>New user?<a href="register_form.php"> Create Account</a></p>
              
   </div>
   </form>

</div>

</body>
</html>