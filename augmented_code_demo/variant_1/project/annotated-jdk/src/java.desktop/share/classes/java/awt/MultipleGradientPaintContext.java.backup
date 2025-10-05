/*
 * Copyright (c) 2006, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.MultipleGradientPaint.CycleMethod;
import java.awt.MultipleGradientPaint.ColorSpaceType;
import java.awt.color.ColorSpace;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Rectangle2D;
import java.awt.image.ColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferInt;
import java.awt.image.DirectColorModel;
import java.awt.image.Raster;
import java.awt.image.SinglePixelPackedSampleModel;
import java.awt.image.WritableRaster;
import java.lang.ref.SoftReference;
import java.lang.ref.WeakReference;
import java.util.Arrays;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
abstract class MultipleGradientPaintContext implements PaintContext {

    protected ColorModel model;

    protected static ColorModel cachedModel;

    protected static WeakReference<Raster> cached;

    protected Raster saved;

    protected CycleMethod cycleMethod;

    protected ColorSpaceType colorSpace;

    protected float a00, a01, a10, a11, a02, a12;

    protected boolean isSimpleLookup;

    protected int fastGradientArraySize;

    protected int[] gradient;

    protected static final int GRADIENT_SIZE;

    protected static final int GRADIENT_SIZE_INDEX;

    protected MultipleGradientPaintContext(MultipleGradientPaint mgp, ColorModel cm, Rectangle deviceBounds, Rectangle2D userBounds, AffineTransform t, RenderingHints hints, float[] fractions, Color[] colors, CycleMethod cycleMethod, ColorSpaceType colorSpace) {
    }

    protected final int indexIntoGradientsArrays(float position);

    public final Raster getRaster(int x, int y, int w, int h);

    protected abstract void fillRaster(int[] pixels, int off, int adjust, int x, int y, int w, int h);

    public final void dispose();

    public final ColorModel getColorModel();
}
