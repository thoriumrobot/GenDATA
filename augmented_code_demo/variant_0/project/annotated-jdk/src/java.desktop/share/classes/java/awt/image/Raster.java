/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.awt.image;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Point;
    @Positive
import sun.awt.image.ByteInterleavedRaster;
    @Positive
import sun.awt.image.ShortInterleavedRaster;
    @Positive
import sun.awt.image.IntegerInterleavedRaster;
    @Positive
import sun.awt.image.ByteBandedRaster;
    @Positive
import sun.awt.image.ShortBandedRaster;
    @Positive
import sun.awt.image.BytePackedRaster;
    @Positive
import sun.awt.image.SunWritableRaster;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Raster {

    @Positive
    protected SampleModel sampleModel;

    @Positive
    protected DataBuffer dataBuffer;

    @Positive
    protected int minX;

    @Positive
    protected int minY;

    @Positive
    protected int width;

    @Positive
    protected int height;

    @Positive
    protected int sampleModelTranslateX;

    @Positive
    protected int sampleModelTranslateY;

    @Positive
    protected int numBands;

    @Positive
    protected int numDataElements;

    @Positive
    protected Raster parent;

    @Positive
    public static WritableRaster createInterleavedRaster(int dataType, int w, int h, int bands, Point location);

    @Positive
    public static WritableRaster createInterleavedRaster(int dataType, int w, int h, int scanlineStride, int pixelStride, int[] bandOffsets, Point location);

    @Positive
    public static WritableRaster createBandedRaster(int dataType, int w, int h, int bands, Point location);

    @Positive
    public static WritableRaster createBandedRaster(int dataType, int w, int h, int scanlineStride, int[] bankIndices, int[] bandOffsets, Point location);

    @Positive
    public static WritableRaster createPackedRaster(int dataType, int w, int h, int[] bandMasks, Point location);

    @Positive
    public static WritableRaster createPackedRaster(int dataType, int w, int h, int bands, int bitsPerBand, Point location);

    @Positive
    public static WritableRaster createInterleavedRaster(DataBuffer dataBuffer, int w, int h, int scanlineStride, int pixelStride, int[] bandOffsets, Point location);

    @Positive
    public static WritableRaster createBandedRaster(DataBuffer dataBuffer, int w, int h, int scanlineStride, int[] bankIndices, int[] bandOffsets, Point location);

    @Positive
    public static WritableRaster createPackedRaster(DataBuffer dataBuffer, int w, int h, int scanlineStride, int[] bandMasks, Point location);

    @Positive
    public static WritableRaster createPackedRaster(DataBuffer dataBuffer, int w, int h, int bitsPerPixel, Point location);

    @Positive
    public static Raster createRaster(SampleModel sm, DataBuffer db, Point location);

    @Positive
    public static WritableRaster createWritableRaster(SampleModel sm, Point location);

    @Positive
    public static WritableRaster createWritableRaster(SampleModel sm, DataBuffer db, Point location);

    @Positive
    protected Raster(SampleModel sampleModel, Point origin) {
    @Positive
    }

    @Positive
    protected Raster(SampleModel sampleModel, DataBuffer dataBuffer, Point origin) {
    @Positive
    }

    @Positive
    protected Raster(SampleModel sampleModel, DataBuffer dataBuffer, Rectangle aRegion, Point sampleModelTranslate, Raster parent) {
    @Positive
    }

    @Positive
    public Raster getParent();

    @Positive
    public final int getSampleModelTranslateX();

    @Positive
    public final int getSampleModelTranslateY();

    @Positive
    public WritableRaster createCompatibleWritableRaster();

    @Positive
    public WritableRaster createCompatibleWritableRaster(int w, int h);

    @Positive
    public WritableRaster createCompatibleWritableRaster(Rectangle rect);

    @Positive
    public WritableRaster createCompatibleWritableRaster(int x, int y, int w, int h);

    @Positive
    public Raster createTranslatedChild(int childMinX, int childMinY);

    @Positive
    public Raster createChild(int parentX, int parentY, int width, int height, int childMinX, int childMinY, int[] bandList);

    @Positive
    public Rectangle getBounds();

    @Positive
    public final int getMinX();

    @Positive
    public final int getMinY();

    @Positive
    public final int getWidth();

    @Positive
    public final int getHeight();

    @Positive
    public final int getNumBands();

    @Positive
    public final int getNumDataElements();

    @Positive
    public final int getTransferType();

    @Positive
    public DataBuffer getDataBuffer();

    @Positive
    public SampleModel getSampleModel();

    @Positive
    public Object getDataElements(int x, int y, Object outData);

    @Positive
    public Object getDataElements(int x, int y, int w, int h, Object outData);

    @Positive
    public int[] getPixel(int x, int y, int[] iArray);

    @Positive
    public float[] getPixel(int x, int y, float[] fArray);

    @Positive
    public double[] getPixel(int x, int y, double[] dArray);

    @Positive
    public int[] getPixels(int x, int y, int w, int h, int[] iArray);

    @Positive
    public float[] getPixels(int x, int y, int w, int h, float[] fArray);

    @Positive
    public double[] getPixels(int x, int y, int w, int h, double[] dArray);

    @Positive
    public int getSample(int x, int y, int b);

    @Positive
    public float getSampleFloat(int x, int y, int b);

    @Positive
    public double getSampleDouble(int x, int y, int b);

    @Positive
    public int[] getSamples(int x, int y, int w, int h, int b, int[] iArray);

    @Positive
    public float[] getSamples(int x, int y, int w, int h, int b, float[] fArray);

    @Positive
    public double[] getSamples(int x, int y, int w, int h, int b, double[] dArray);
    @Positive
}

// CFWR semantic augmentation - variant 0
