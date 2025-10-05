/*
    @Positive
 * Copyright (c) 2006, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.MultipleGradientPaint.CycleMethod;
    @Positive
import java.awt.MultipleGradientPaint.ColorSpaceType;
    @Positive
import java.awt.color.ColorSpace;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.NoninvertibleTransformException;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.awt.image.DataBuffer;
    @Positive
import java.awt.image.DataBufferInt;
    @Positive
import java.awt.image.DirectColorModel;
    @Positive
import java.awt.image.Raster;
    @Positive
import java.awt.image.SinglePixelPackedSampleModel;
    @Positive
import java.awt.image.WritableRaster;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.util.Arrays;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
abstract class MultipleGradientPaintContext implements PaintContext {

    @Positive
    protected ColorModel model;

    @Positive
    protected static ColorModel cachedModel;

    @Positive
    protected static WeakReference<Raster> cached;

    @Positive
    protected Raster saved;

    @Positive
    protected CycleMethod cycleMethod;

    @Positive
    protected ColorSpaceType colorSpace;

    @Positive
    protected float a00, a01, a10, a11, a02, a12;

    @Positive
    protected boolean isSimpleLookup;

    @Positive
    protected int fastGradientArraySize;

    @Positive
    protected int[] gradient;

    @Positive
    protected static final int GRADIENT_SIZE;

    @Positive
    protected static final int GRADIENT_SIZE_INDEX;

    @Positive
    protected MultipleGradientPaintContext(MultipleGradientPaint mgp, ColorModel cm, Rectangle deviceBounds, Rectangle2D userBounds, AffineTransform t, RenderingHints hints, float[] fractions, Color[] colors, CycleMethod cycleMethod, ColorSpaceType colorSpace) {
    @Positive
    }

    @Positive
    protected final int indexIntoGradientsArrays(float position);

    @Positive
    public final Raster getRaster(int x, int y, int w, int h);

    @Positive
    protected abstract void fillRaster(int[] pixels, int off, int adjust, int x, int y, int w, int h);

    @Positive
    public final void dispose();

    @Positive
    public final ColorModel getColorModel();
    @Positive
}

// CFWR semantic augmentation - variant 0
