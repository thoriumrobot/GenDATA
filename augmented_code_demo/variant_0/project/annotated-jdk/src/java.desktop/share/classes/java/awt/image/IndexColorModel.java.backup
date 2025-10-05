/*
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.awt.image;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.math.BigInteger;
import java.util.Arrays;

public class IndexColorModel extends ColorModel {

    public IndexColorModel(int bits, int size, byte[] r, byte[] g, byte[] b) {
    }

    public IndexColorModel(int bits, int size, byte[] r, byte[] g, byte[] b, int trans) {
    }

    public IndexColorModel(int bits, int size, byte[] r, byte[] g, byte[] b, byte[] a) {
    }

    public IndexColorModel(int bits, int size, byte[] cmap, int start, boolean hasalpha) {
    }

    public IndexColorModel(int bits, int size, byte[] cmap, int start, boolean hasalpha, int trans) {
    }

    public IndexColorModel(int bits, int size, int[] cmap, int start, boolean hasalpha, int trans, int transferType) {
    }

    public IndexColorModel(int bits, int size, int[] cmap, int start, int transferType, BigInteger validBits) {
    }

    public int getTransparency();

    public int[] getComponentSize();

    public final int getMapSize();

    public final int getTransparentPixel();

    public final void getReds(byte[] r);

    public final void getGreens(byte[] g);

    public final void getBlues(byte[] b);

    public final void getAlphas(byte[] a);

    public final void getRGBs(int[] rgb);

    public final int getRed(int pixel);

    public final int getGreen(int pixel);

    public final int getBlue(int pixel);

    public final int getAlpha(int pixel);

    public final int getRGB(int pixel);

    public synchronized Object getDataElements(int rgb, Object pixel);

    public int[] getComponents(int pixel, int[] components, int offset);

    public int[] getComponents(Object pixel, int[] components, int offset);

    public int getDataElement(int[] components, int offset);

    public Object getDataElements(int[] components, int offset, Object pixel);

    public WritableRaster createCompatibleWritableRaster(int w, int h);

    public boolean isCompatibleRaster(Raster raster);

    public SampleModel createCompatibleSampleModel(int w, int h);

    public boolean isCompatibleSampleModel(SampleModel sm);

    public BufferedImage convertToIntDiscrete(Raster raster, boolean forceARGB);

    public boolean isValid(int pixel);

    public boolean isValid();

    public BigInteger getValidPixels();

    @Deprecated()
    @SuppressWarnings("removal")
    public void finalize();

    public String toString();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();
}
