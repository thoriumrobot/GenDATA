/*
 * Copyright (c) 1997, 2018, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.color.ColorSpace;
import java.awt.geom.Rectangle2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.geom.Point2D;
import sun.awt.image.ImagingLib;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class LookupOp implements BufferedImageOp, RasterOp {

    public LookupOp(LookupTable lookup, RenderingHints hints) {
    }

    public final LookupTable getTable();

    public final BufferedImage filter(BufferedImage src, BufferedImage dst);

    public final WritableRaster filter(Raster src, WritableRaster dst);

    public final Rectangle2D getBounds2D(BufferedImage src);

    public final Rectangle2D getBounds2D(Raster src);

    public BufferedImage createCompatibleDestImage(BufferedImage src, ColorModel destCM);

    public WritableRaster createCompatibleDestRaster(Raster src);

    public final Point2D getPoint2D(Point2D srcPt, Point2D dstPt);

    public final RenderingHints getRenderingHints();
}
