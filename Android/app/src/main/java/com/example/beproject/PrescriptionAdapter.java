package com.example.beproject;

import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.squareup.picasso.Callback;
import com.squareup.picasso.NetworkPolicy;
import com.squareup.picasso.Picasso;

import org.jetbrains.annotations.NotNull;

import java.util.List;

import de.hdodenhof.circleimageview.CircleImageView;

public class PrescriptionAdapter extends RecyclerView.Adapter<PrescriptionAdapter.PrescriptionViewHolder>{
    Context context;
    private FirebaseAuth mAuth;
    private List<Prescription> mMessageList;
    private DatabaseReference mUserDatabase;

    public PrescriptionAdapter(List<Prescription> mMessageList) {
        this.mMessageList = mMessageList;
    }

    @Override
    public PrescriptionViewHolder onCreateViewHolder(ViewGroup parent, int viewType){

        mAuth=FirebaseAuth.getInstance();
        View v= LayoutInflater.from(parent.getContext()).inflate(R.layout.list,parent,false);
        return new PrescriptionViewHolder(v);
    }

    public class PrescriptionViewHolder extends RecyclerView.ViewHolder {

        public TextView date;
        public View mView;

        public PrescriptionViewHolder(@NonNull View itemView) {
            super(itemView);

            date=itemView.findViewById(R.id.date);
            mView = itemView.findViewById(R.id.list);

        }
    }

    @Override
    public void onBindViewHolder(@NonNull final PrescriptionViewHolder holder, int position) {

        String current_user_id=mAuth.getCurrentUser().getUid();
        mUserDatabase= FirebaseDatabase.getInstance().getReference().child("Users");
        mUserDatabase.keepSynced(true);

        final Prescription c=mMessageList.get(position);

        String date=c.getDate();
        String imageUri=c.getImageUri();

        holder.date.setText(date);
        holder.mView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent displayImg = new Intent(view.getContext(),ImageDisplayActivity.class);
                displayImg.putExtra("imageUri",imageUri);
                view.getContext().startActivity(displayImg);
            }
        });


    }

    @Override
    public int getItemCount() {
        return mMessageList.size();
    }

}
