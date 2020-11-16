package com.example.furnitureclassifier

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private var bitmap:Bitmap? = null
    private var module:Module? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        try {
            bitmap = BitmapFactory.decodeStream(assets.open("sofa_image.jpg"))

            // Loading serialized torchscript module
            module = Module.load(assetFilePath(this, "furniture_mobile_model.pt"))
            Log.d("Model", "Model Loaded Successfully")
        } catch (e : IOException) {
            Log.e("FurnitureClassifier", "Error reading assets", e)
        }

        // Showing on UI
        val imageView: ImageView = findViewById(R.id.image)
        imageView.setImageBitmap(bitmap)

        // preparing input tensor
        val inputTensor: Tensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB)

        val outputTensor : Tensor = module!!.forward(IValue.from(inputTensor)).toTensor()

        // getting tensor content as array of floats
        val scores = outputTensor.dataAsFloatArray
        // Searching for the index with maximum score
        var maxScore = -Float.MAX_VALUE
        var maxScoreIdx = -1

        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxScoreIdx = i
            }
        }

        val className = FurnitureClasses.FURNITURE_CLASS[maxScoreIdx]

        // Showing className on UI
        val textView: TextView = findViewById(R.id.text)
        textView.text = "predicted image as $className"
    }

    @Throws(IOException::class)
    fun assetFilePath(context: Context, assetName: String?): String? {
        val file = File(context.filesDir, assetName!!)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { `is` ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
            return file.absolutePath
        }
    }
}

