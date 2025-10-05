/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, Oracle and/or its affiliates. All rights reserved.
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
package sun.font;

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
import java.awt.FontFormatException;
    @Positive
import java.awt.geom.GeneralPath;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.io.FileInputStream;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.channels.FileChannel;

    @Positive
public abstract class PhysicalFont extends Font2D {

    @Positive
    protected String platName;

    @Positive
    protected Object nativeNames;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();

    @Positive
    protected PhysicalFont() {
    @Positive
    }

    @Positive
    Point2D.Float getGlyphPoint(long pScalerContext, int glyphCode, int ptNumber);

    @Positive
    abstract StrikeMetrics getFontMetrics(long pScalerContext);

    @Positive
    abstract float getGlyphAdvance(long pScalerContext, int glyphCode);

    @Positive
    abstract void getGlyphMetrics(long pScalerContext, int glyphCode, Point2D.Float metrics);

    @Positive
    abstract long getGlyphImage(long pScalerContext, int glyphCode);

    @Positive
    abstract Rectangle2D.Float getGlyphOutlineBounds(long pScalerContext, int glyphCode);

    @Positive
    abstract GeneralPath getGlyphOutline(long pScalerContext, int glyphCode, float x, float y);

    @Positive
    abstract GeneralPath getGlyphVectorOutline(long pScalerContext, int[] glyphs, int numGlyphs, float x, float y);
    @Positive
}
