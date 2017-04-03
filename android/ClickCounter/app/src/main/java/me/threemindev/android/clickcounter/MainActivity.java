package me.threemindev.android.clickcounter;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    private int clickCount=2;
    Button clickButton;
    TextView textViewButton;

    public void updateClickCountTextView() {
        textViewButton.setText("횟수 : " + clickCount);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        clickButton = (Button) findViewById(R.id.clickButton);
        textViewButton = (TextView) findViewById(R.id.clickCountTextView);

        updateClickCountTextView();
        clickButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                clickCount += 1;
                updateClickCountTextView();
            }
        });
    }
}
