/*
 * Copyright (c) 1998, 2017, Oracle and/or its affiliates. All rights reserved.
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
package sun.font;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import static java.awt.RenderingHints.*;
import java.awt.Shape;
import java.awt.font.FontRenderContext;
import java.awt.font.GlyphMetrics;
import java.awt.font.GlyphJustificationInfo;
import java.awt.font.GlyphVector;
import java.awt.font.LineMetrics;
import java.awt.font.TextAttribute;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.PathIterator;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.lang.ref.SoftReference;
import java.text.CharacterIterator;
import sun.awt.SunHints;
import sun.java2d.loops.FontInfo;

public class StandardGlyphVector extends GlyphVector {

    public StandardGlyphVector(Font font, String str, FontRenderContext frc) {
    }

    public StandardGlyphVector(Font font, char[] text, FontRenderContext frc) {
    }

    public StandardGlyphVector(Font font, char[] text, int start, int count, FontRenderContext frc) {
    }

    public StandardGlyphVector(Font font, FontRenderContext frc, int[] glyphs, float[] positions, int[] indices, int flags) {
    }

    public void initGlyphVector(Font font, FontRenderContext frc, int[] glyphs, float[] positions, int[] indices, int flags);

    public StandardGlyphVector(Font font, CharacterIterator iter, FontRenderContext frc) {
    }

    public StandardGlyphVector(Font font, int[] glyphs, FontRenderContext frc) {
    }

    public static StandardGlyphVector getStandardGV(GlyphVector gv, FontInfo info);

    public Font getFont();

    public FontRenderContext getFontRenderContext();

    public void performDefaultLayout();

    public int getNumGlyphs();

    public int getGlyphCode(int glyphIndex);

    public int[] getGlyphCodes(int start, int count, int[] result);

    public int getGlyphCharIndex(int ix);

    public int[] getGlyphCharIndices(int start, int count, int[] result);

    public Rectangle2D getLogicalBounds();

    public Rectangle2D getVisualBounds();

    public Rectangle getPixelBounds(FontRenderContext renderFRC, float x, float y);

    public Shape getOutline();

    public Shape getOutline(float x, float y);

    public Shape getGlyphOutline(int ix);

    public Shape getGlyphOutline(int ix, float x, float y);

    public Point2D getGlyphPosition(int ix);

    public void setGlyphPosition(int ix, Point2D pos);

    public AffineTransform getGlyphTransform(int ix);

    public void setGlyphTransform(int ix, AffineTransform newTX);

    public int getLayoutFlags();

    public float[] getGlyphPositions(int start, int count, float[] result);

    public Shape getGlyphLogicalBounds(int ix);

    public Shape getGlyphVisualBounds(int ix);

    public Rectangle getGlyphPixelBounds(int index, FontRenderContext renderFRC, float x, float y);

    public GlyphMetrics getGlyphMetrics(int ix);

    public GlyphJustificationInfo getGlyphJustificationInfo(int ix);

    public boolean equals(GlyphVector rhs);

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object rhs);

    public StandardGlyphVector copy();

    public Object clone();

    public void setGlyphPositions(float[] srcPositions, int srcStart, int start, int count);

    public void setGlyphPositions(float[] srcPositions);

    public float[] getGlyphPositions(float[] result);

    public AffineTransform[] getGlyphTransforms(int start, int count, AffineTransform[] result);

    public AffineTransform[] getGlyphTransforms();

    public void setGlyphTransforms(AffineTransform[] srcTransforms, int srcStart, int start, int count);

    public void setGlyphTransforms(AffineTransform[] srcTransforms);

    public float[] getGlyphInfo();

    boolean needsPositions(double[] devTX);

    Object setupGlyphImages(long[] images, float[] positions, double[] devTX);

    int[] getValidatedGlyphs(int[] oglyphs);

    public static final int FLAG_USES_VERTICAL_BASELINE;

    public static final int FLAG_USES_VERTICAL_METRICS;

    public static final int FLAG_USES_ALTERNATE_ORIENTATION;

    static final class GlyphTransformInfo {

        public boolean equals(GlyphTransformInfo rhs);

        void setGlyphTransform(int glyphIndex, AffineTransform newTX);

        AffineTransform getGlyphTransform(int ix);

        int transformCount();

        Object setupGlyphImages(long[] images, float[] positions, AffineTransform tx);

        Rectangle getGlyphsPixelBounds(AffineTransform tx, float x, float y, int start, int count);

        GlyphStrike getStrike(int glyphIndex);
    }

    public static final class GlyphStrike {

        static GlyphStrike create(StandardGlyphVector sgv, AffineTransform dtx, AffineTransform gtx);

        void getADL(ADL result);

        void getGlyphPosition(int glyphID, int ix, float[] positions, float[] result);

        void addDefaultGlyphAdvance(int glyphID, Point2D.Float result);

        Rectangle2D getGlyphOutlineBounds(int glyphID, float x, float y);

        void appendGlyphOutline(int glyphID, GeneralPath result, float x, float y);
    }

    public String toString();

    StringBuffer appendString(StringBuffer buf);

    static class ADL {

        public float ascentX;

        public float ascentY;

        public float descentX;

        public float descentY;

        public float leadingX;

        public float leadingY;

        public String toString();

        protected StringBuffer toStringBuffer(StringBuffer result);
    }
}
