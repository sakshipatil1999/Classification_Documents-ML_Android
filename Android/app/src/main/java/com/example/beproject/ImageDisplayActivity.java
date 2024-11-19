package com.example.beproject;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.ImageView;

import com.squareup.picasso.Callback;
import com.squareup.picasso.NetworkPolicy;
import com.squareup.picasso.Picasso;

public class ImageDisplayActivity extends AppCompatActivity {
    String imageUri;
    ImageView image;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_display);

        imageUri=getIntent().getStringExtra("imageUri");

        image=(ImageView) findViewById(R.id.image);
        Picasso.with(getBaseContext()).load(imageUri).networkPolicy(NetworkPolicy.OFFLINE)
                .placeholder(R.drawable.demo1).into(image, new Callback() {
            @Override
            public void onSuccess() {

            }

            @Override
            public void onError() {
                Picasso.with(getBaseContext()).load(imageUri)
                        .placeholder(R.drawable.demo1).into(image);
            }
        });
    }
}