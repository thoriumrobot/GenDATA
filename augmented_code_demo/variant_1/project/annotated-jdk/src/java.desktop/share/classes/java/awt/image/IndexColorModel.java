/*
    @Positive
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.awt.Transparency;
    @Positive
import java.awt.color.ColorSpace;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.Arrays;

    @Positive
public class IndexColorModel extends ColorModel {

    @Positive
    public IndexColorModel(int bits, int size, byte[] r, byte[] g, byte[] b) {
    @Positive
    }

    @Positive
    public IndexColorModel(int bits, int size, byte[] r, byte[] g, byte[] b, int trans) {
    @Positive
    }

    @Positive
    public IndexColorModel(int bits, int size, byte[] r, byte[] g, byte[] b, byte[] a) {
    @Positive
    }

    @Positive
    public IndexColorModel(int bits, int size, byte[] cmap, int start, boolean hasalpha) {
    @Positive
    }

    @Positive
    public IndexColorModel(int bits, int size, byte[] cmap, int start, boolean hasalpha, int trans) {
    @Positive
    }

    @Positive
    public IndexColorModel(int bits, int size, int[] cmap, int start, boolean hasalpha, int trans, int transferType) {
    @Positive
    }

    @Positive
    public IndexColorModel(int bits, int size, int[] cmap, int start, int transferType, BigInteger validBits) {
    @Positive
    }

    @Positive
    public int getTransparency();

    @Positive
    public int[] getComponentSize();

    @Positive
    public final int getMapSize();

    @Positive
    public final int getTransparentPixel();

    @Positive
    public final void getReds(byte[] r);

    @Positive
    public final void getGreens(byte[] g);

    @Positive
    public final void getBlues(byte[] b);

    @Positive
    public final void getAlphas(byte[] a);

    @Positive
    public final void getRGBs(int[] rgb);

    @Positive
    public final int getRed(int pixel);

    @Positive
    public final int getGreen(int pixel);

    @Positive
    public final int getBlue(int pixel);

    @Positive
    public final int getAlpha(int pixel);

    @Positive
    public final int getRGB(int pixel);

    @Positive
    public synchronized Object getDataElements(int rgb, Object pixel);

    @Positive
    public int[] getComponents(int pixel, int[] components, int offset);

    @Positive
    public int[] getComponents(Object pixel, int[] components, int offset);

    @Positive
    public int getDataElement(int[] components, int offset);

    @Positive
    public Object getDataElements(int[] components, int offset, Object pixel);

    @Positive
    public WritableRaster createCompatibleWritableRaster(int w, int h);

    @Positive
    public boolean isCompatibleRaster(Raster raster);

    @Positive
    public SampleModel createCompatibleSampleModel(int w, int h);

    @Positive
    public boolean isCompatibleSampleModel(SampleModel sm);

    @Positive
    public BufferedImage convertToIntDiscrete(Raster raster, boolean forceARGB);

    @Positive
    public boolean isValid(int pixel);

    @Positive
    public boolean isValid();

    @Positive
    public BigInteger getValidPixels();

    @Positive
    @Deprecated()
    @Positive
    @SuppressWarnings("removal")
    @Positive
    public void finalize();

    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 1
