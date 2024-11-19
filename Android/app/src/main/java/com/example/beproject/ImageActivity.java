package com.example.beproject;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ServerValue;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import de.hdodenhof.circleimageview.CircleImageView;

public class ImageActivity extends AppCompatActivity {
    TextView textView;
    TextView textView1;
    ImageView iv,iv2;
    BitmapDrawable drawable;
    Bitmap bitmap;
    String imgString="";
    TessOCR mTessOCR;
    Button bAddButton;
    Button bSaveButton;
    private StorageReference mImageStorage;
    private DatabaseReference mDatabase;
    private FirebaseAuth mAuth;
    private Uri selectedImage;


    public static final String lang = "eng";
    public static final String DATA_PATH = "/mnt/sdcard/tesseract/";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image);

        mTessOCR = new TessOCR (this, lang);
        textView = (TextView) findViewById(R.id.text);
        textView1 = (TextView) findViewById(R.id.category);
        bAddButton = (Button) findViewById(R.id.add_button);
        bSaveButton = (Button) findViewById(R.id.save_button);
        iv=(ImageView) findViewById(R.id.imageView);
        mAuth=FirebaseAuth.getInstance();
        mDatabase= FirebaseDatabase.getInstance().getReference();
        mImageStorage= FirebaseStorage.getInstance().getReference();

        bAddButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent pickPhoto = new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(pickPhoto , 1);//one can be replaced with any action code

            }
        });

        bSaveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
               // save();
                AlertDialog.Builder builder = new AlertDialog.Builder(view.getContext());

                builder.setTitle("Confirm");
                builder.setMessage("Do you want to save the image to " + textView1.getText().toString()+"?");

                builder.setPositiveButton("YES", new DialogInterface.OnClickListener() {

                    public void onClick(DialogInterface dialog, int which) {
                        // Do nothing but close the dialog
                        DatabaseReference user_message_push=mDatabase.child("Users").child(mAuth.getCurrentUser().getUid()).child(textView1.getText().toString()).push();
                        save(user_message_push);
                        dialog.dismiss();
                    }
                });

                builder.setNegativeButton("NO", new DialogInterface.OnClickListener() {

                    @Override
                    public void onClick(DialogInterface dialog, int which) {

                        if(textView1.getText().toString().equalsIgnoreCase("Prescription")) {
                            DatabaseReference user_message_push = mDatabase.child("Users").child(mAuth.getCurrentUser().getUid()).child("Report").push();
                            save(user_message_push);
                        }else{
                            DatabaseReference user_message_push=mDatabase.child("Users").child(mAuth.getCurrentUser().getUid()).child("Prescription").push();
                            save(user_message_push);
                        }

                        // Do nothing
                        dialog.dismiss();
                    }
                });

                AlertDialog alert = builder.create();
                alert.show();
            }
        });


    }

    private String getStringImage(Bitmap bitmap){
        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        bitmap.compress(Bitmap.CompressFormat.PNG,100,baos);
        byte[] imagebytes= baos.toByteArray();
        String encodedImage = android.util.Base64.encodeToString(imagebytes, Base64.DEFAULT);
        return encodedImage;
    }

    private void selectImage(Context context) {
        final CharSequence[] options = { "Take Photo", "Choose from Gallery","Cancel" };

        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle("Choose your profile picture");

        builder.setItems(options, new DialogInterface.OnClickListener() {

            @Override
            public void onClick(DialogInterface dialog, int item) {

                if (options[item].equals("Take Photo")) {
                    Intent takePicture = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(takePicture, 0);//zero can be replaced with any action code (called requestCode)

                } else if (options[item].equals("Choose from Gallery")) {
                    Intent pickPhoto = new Intent(Intent.ACTION_PICK,
                            android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(pickPhoto , 1);//one can be replaced with any action code

                } else if (options[item].equals("Cancel")) {
                    dialog.dismiss();
                }
            }
        });
        builder.show();
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent imageReturnedIntent) {
        super.onActivityResult(requestCode, resultCode, imageReturnedIntent);
        switch(requestCode) {
            case 0:
                if(resultCode == RESULT_OK){
                    selectedImage = imageReturnedIntent.getData();
                    iv.setImageURI(selectedImage);
                }

                break;
            case 1:
                if(resultCode == RESULT_OK){
                    selectedImage = imageReturnedIntent.getData();
                    iv.setImageURI(selectedImage);
                    recognize();
                }
                break;
        }
    }

    private void recognize(){

        Python py = Python.getInstance();
        drawable = (BitmapDrawable) iv.getDrawable();
        if(drawable != null) {
            bitmap = drawable.getBitmap();
            imgString = getStringImage(bitmap);

            PyObject pyobj = py.getModule("image");

            PyObject obj = pyobj.callAttr("main", imgString);

            String str = obj.toString();
            byte data[] = android.util.Base64.decode(str, Base64.DEFAULT);

            Bitmap bmp = BitmapFactory.decodeByteArray(data, 0, data.length);
            // iv2.setImageBitmap(bmp);
            final String result = mTessOCR.getOCRResult(bmp);
           // textView.setText(result);

            PyObject pyobj1 = py.getModule("pyscript");

            PyObject obj1 = pyobj1.callAttr("main", result);
            textView1.setText(obj1.toString());


        }
    }

    private void save(DatabaseReference user_message_push){
       // DatabaseReference user_message_push=mDatabase.child("Users").child(mAuth.getCurrentUser().getUid()).child(textView1.getText().toString()).push();

        final String push_id =user_message_push.getKey();

        final StorageReference filepath=mImageStorage.child("images").child(push_id+".jpg");
        Date d1 = new Date();
        filepath.putFile(selectedImage).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
            @Override
            public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                filepath.getDownloadUrl().addOnSuccessListener(new OnSuccessListener<Uri>() {
                    @Override
                    public void onSuccess(Uri uri) {

                        String download_url=uri.toString();
                        HashMap<String,String> messageMap =new HashMap<>();
                        messageMap.put("imageUri",download_url);
                        messageMap.put("date",d1.toString());

                        user_message_push.setValue(messageMap).addOnSuccessListener(new OnSuccessListener<Void>() {
                            @Override
                            public void onSuccess(Void aVoid) {
                                // Sign in success, update UI with the signed-in user's information

                                Toast.makeText(getBaseContext(), "User created.",
                                        Toast.LENGTH_SHORT).show();

                                Intent mainIntent = new Intent(getBaseContext(),MainActivity.class);
                                startActivity(mainIntent);
                                finish();
                            }
                        });



                    }
                });
            }
        });

    }


}