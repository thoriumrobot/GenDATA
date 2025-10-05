/*
    @Positive
 * Copyright (c) 1998, 2017, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.Font;
    @Positive
import java.awt.Graphics2D;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.Rectangle;
    @Positive
import static java.awt.RenderingHints.*;
    @Positive
import java.awt.Shape;
    @Positive
import java.awt.font.FontRenderContext;
    @Positive
import java.awt.font.GlyphMetrics;
    @Positive
import java.awt.font.GlyphJustificationInfo;
    @Positive
import java.awt.font.GlyphVector;
    @Positive
import java.awt.font.LineMetrics;
    @Positive
import java.awt.font.TextAttribute;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.GeneralPath;
    @Positive
import java.awt.geom.NoninvertibleTransformException;
    @Positive
import java.awt.geom.PathIterator;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.text.CharacterIterator;
    @Positive
import sun.awt.SunHints;
    @Positive
import sun.java2d.loops.FontInfo;

    @Positive
public class StandardGlyphVector extends GlyphVector {

    @Positive
    public StandardGlyphVector(Font font, String str, FontRenderContext frc) {
    @Positive
    }

    @Positive
    public StandardGlyphVector(Font font, char[] text, FontRenderContext frc) {
    @Positive
    }

    @Positive
    public StandardGlyphVector(Font font, char[] text, int start, int count, FontRenderContext frc) {
    @Positive
    }

    @Positive
    public StandardGlyphVector(Font font, FontRenderContext frc, int[] glyphs, float[] positions, int[] indices, int flags) {
    @Positive
    }

    @Positive
    public void initGlyphVector(Font font, FontRenderContext frc, int[] glyphs, float[] positions, int[] indices, int flags);

    @Positive
    public StandardGlyphVector(Font font, CharacterIterator iter, FontRenderContext frc) {
    @Positive
    }

    @Positive
    public StandardGlyphVector(Font font, int[] glyphs, FontRenderContext frc) {
    @Positive
    }

    @Positive
    public static StandardGlyphVector getStandardGV(GlyphVector gv, FontInfo info);

    @Positive
    public Font getFont();

    @Positive
    public FontRenderContext getFontRenderContext();

    @Positive
    public void performDefaultLayout();

    @Positive
    public int getNumGlyphs();

    @Positive
    public int getGlyphCode(int glyphIndex);

    @Positive
    public int[] getGlyphCodes(int start, int count, int[] result);

    @Positive
    public int getGlyphCharIndex(int ix);

    @Positive
    public int[] getGlyphCharIndices(int start, int count, int[] result);

    @Positive
    public Rectangle2D getLogicalBounds();

    @Positive
    public Rectangle2D getVisualBounds();

    @Positive
    public Rectangle getPixelBounds(FontRenderContext renderFRC, float x, float y);

    @Positive
    public Shape getOutline();

    @Positive
    public Shape getOutline(float x, float y);

    @Positive
    public Shape getGlyphOutline(int ix);

    @Positive
    public Shape getGlyphOutline(int ix, float x, float y);

    @Positive
    public Point2D getGlyphPosition(int ix);

    @Positive
    public void setGlyphPosition(int ix, Point2D pos);

    @Positive
    public AffineTransform getGlyphTransform(int ix);

    @Positive
    public void setGlyphTransform(int ix, AffineTransform newTX);

    @Positive
    public int getLayoutFlags();

    @Positive
    public float[] getGlyphPositions(int start, int count, float[] result);

    @Positive
    public Shape getGlyphLogicalBounds(int ix);

    @Positive
    public Shape getGlyphVisualBounds(int ix);

    @Positive
    public Rectangle getGlyphPixelBounds(int index, FontRenderContext renderFRC, float x, float y);

    @Positive
    public GlyphMetrics getGlyphMetrics(int ix);

    @Positive
    public GlyphJustificationInfo getGlyphJustificationInfo(int ix);

    @Positive
    public boolean equals(GlyphVector rhs);

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object rhs);

    @Positive
    public StandardGlyphVector copy();

    @Positive
    public Object clone();

    @Positive
    public void setGlyphPositions(float[] srcPositions, int srcStart, int start, int count);

    @Positive
    public void setGlyphPositions(float[] srcPositions);

    @Positive
    public float[] getGlyphPositions(float[] result);

    @Positive
    public AffineTransform[] getGlyphTransforms(int start, int count, AffineTransform[] result);

    @Positive
    public AffineTransform[] getGlyphTransforms();

    @Positive
    public void setGlyphTransforms(AffineTransform[] srcTransforms, int srcStart, int start, int count);

    @Positive
    public void setGlyphTransforms(AffineTransform[] srcTransforms);

    @Positive
    public float[] getGlyphInfo();

    @Positive
    boolean needsPositions(double[] devTX);

    @Positive
    Object setupGlyphImages(long[] images, float[] positions, double[] devTX);

    @Positive
    int[] getValidatedGlyphs(int[] oglyphs);

    @Positive
    public static final int FLAG_USES_VERTICAL_BASELINE;

    @Positive
    public static final int FLAG_USES_VERTICAL_METRICS;

    @Positive
    public static final int FLAG_USES_ALTERNATE_ORIENTATION;

    @Positive
    static final class GlyphTransformInfo {

    @Positive
        public boolean equals(GlyphTransformInfo rhs);

    @Positive
        void setGlyphTransform(int glyphIndex, AffineTransform newTX);

    @Positive
        AffineTransform getGlyphTransform(int ix);

    @Positive
        int transformCount();

    @Positive
        Object setupGlyphImages(long[] images, float[] positions, AffineTransform tx);

    @Positive
        Rectangle getGlyphsPixelBounds(AffineTransform tx, float x, float y, int start, int count);

    @Positive
        GlyphStrike getStrike(int glyphIndex);
    @Positive
    }

    @Positive
    public static final class GlyphStrike {

    @Positive
        static GlyphStrike create(StandardGlyphVector sgv, AffineTransform dtx, AffineTransform gtx);

    @Positive
        void getADL(ADL result);

    @Positive
        void getGlyphPosition(int glyphID, int ix, float[] positions, float[] result);

    @Positive
        void addDefaultGlyphAdvance(int glyphID, Point2D.Float result);

    @Positive
        Rectangle2D getGlyphOutlineBounds(int glyphID, float x, float y);

    @Positive
        void appendGlyphOutline(int glyphID, GeneralPath result, float x, float y);
    @Positive
    }

    @Positive
    public String toString();

    @Positive
    StringBuffer appendString(StringBuffer buf);

    @Positive
    static class ADL {

    @Positive
        public float ascentX;

    @Positive
        public float ascentY;

    @Positive
        public float descentX;

    @Positive
        public float descentY;

    @Positive
        public float leadingX;

    @Positive
        public float leadingY;

    @Positive
        public String toString();

    @Positive
        protected StringBuffer toStringBuffer(StringBuffer result);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
