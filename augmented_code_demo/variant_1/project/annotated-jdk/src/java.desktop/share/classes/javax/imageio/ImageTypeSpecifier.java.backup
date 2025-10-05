/*
 * Copyright (c) 2000, 2014, Oracle and/or its affiliates. All rights reserved.
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
package javax.imageio;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.Point;
import java.awt.Transparency;
import java.awt.image.BandedSampleModel;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.color.ColorSpace;
import java.awt.image.IndexColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DirectColorModel;
import java.awt.image.MultiPixelPackedSampleModel;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.SinglePixelPackedSampleModel;
import java.awt.image.Raster;
import java.awt.image.RenderedImage;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.util.Hashtable;

public class ImageTypeSpecifier {

    protected ColorModel colorModel;

    protected SampleModel sampleModel;

    public ImageTypeSpecifier(ColorModel colorModel, SampleModel sampleModel) {
    }

    public ImageTypeSpecifier(RenderedImage image) {
    }

    static class Packed extends ImageTypeSpecifier {

        public Packed(ColorSpace colorSpace, int redMask, int greenMask, int blueMask, int alphaMask, int transferType, boolean isAlphaPremultiplied) {
        }
    }

    public static ImageTypeSpecifier createPacked(ColorSpace colorSpace, int redMask, int greenMask, int blueMask, int alphaMask, int transferType, boolean isAlphaPremultiplied);

    static ColorModel createComponentCM(ColorSpace colorSpace, int numBands, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied);

    static class Interleaved extends ImageTypeSpecifier {

        public Interleaved(ColorSpace colorSpace, int[] bandOffsets, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied) {
        }

        public boolean equals(Object o);

        public int hashCode();
    }

    public static ImageTypeSpecifier createInterleaved(ColorSpace colorSpace, int[] bandOffsets, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied);

    static class Banded extends ImageTypeSpecifier {

        public Banded(ColorSpace colorSpace, int[] bankIndices, int[] bandOffsets, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied) {
        }

        public boolean equals(Object o);

        public int hashCode();
    }

    public static ImageTypeSpecifier createBanded(ColorSpace colorSpace, int[] bankIndices, int[] bandOffsets, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied);

    static class Grayscale extends ImageTypeSpecifier {

        public Grayscale(int bits, int dataType, boolean isSigned, boolean hasAlpha, boolean isAlphaPremultiplied) {
        }
    }

    public static ImageTypeSpecifier createGrayscale(int bits, int dataType, boolean isSigned);

    public static ImageTypeSpecifier createGrayscale(int bits, int dataType, boolean isSigned, boolean isAlphaPremultiplied);

    static class Indexed extends ImageTypeSpecifier {

        public Indexed(byte[] redLUT, byte[] greenLUT, byte[] blueLUT, byte[] alphaLUT, int bits, int dataType) {
        }
    }

    public static ImageTypeSpecifier createIndexed(byte[] redLUT, byte[] greenLUT, byte[] blueLUT, byte[] alphaLUT, int bits, int dataType);

    public static ImageTypeSpecifier createFromBufferedImageType(int bufferedImageType);

    public static ImageTypeSpecifier createFromRenderedImage(RenderedImage image);

    public int getBufferedImageType();

    public int getNumComponents();

    public int getNumBands();

    public int getBitsPerBand(int band);

    public SampleModel getSampleModel();

    public SampleModel getSampleModel(int width, int height);

    public ColorModel getColorModel();

    public BufferedImage createBufferedImage(int width, int height);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public int hashCode();
}
