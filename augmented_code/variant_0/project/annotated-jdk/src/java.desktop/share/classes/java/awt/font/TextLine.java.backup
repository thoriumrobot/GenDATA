/*
 * Copyright (c) 1998, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.font;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.im.InputMethodHighlight;
import java.awt.image.BufferedImage;
import java.text.Annotation;
import java.text.AttributedCharacterIterator;
import java.text.AttributedCharacterIterator.Attribute;
import java.text.Bidi;
import java.text.CharacterIterator;
import java.util.Hashtable;
import java.util.Map;
import sun.font.AttributeValues;
import sun.font.BidiUtils;
import sun.font.CodePointIterator;
import sun.font.CoreMetrics;
import sun.font.Decoration;
import sun.font.FontLineMetrics;
import sun.font.FontResolver;
import sun.font.GraphicComponent;
import sun.font.LayoutPathImpl;
import sun.font.LayoutPathImpl.EmptyPath;
import sun.font.LayoutPathImpl.SegmentPathBuilder;
import sun.font.TextLabelFactory;
import sun.font.TextLineComponent;
import java.awt.geom.Line2D;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
final class TextLine {

    static final class TextLineMetrics {

        public final float ascent;

        public final float descent;

        public final float leading;

        public final float advance;

        public TextLineMetrics(float ascent, float descent, float leading, float advance) {
        }
    }

    public TextLine(FontRenderContext frc, TextLineComponent[] components, float[] baselineOffsets, char[] chars, int charsStart, int charsLimit, int[] charLogicalOrder, byte[] charLevels, boolean isDirectionLTR) {
    }

    public Rectangle getPixelBounds(FontRenderContext frc, float x, float y);

    static Rectangle computePixelBounds(BufferedImage im);

    private abstract static class Function {

        abstract float computeFunction(TextLine line, int componentIndex, int indexInArray);
    }

    public int characterCount();

    public boolean isDirectionLTR();

    public TextLineMetrics getMetrics();

    public int visualToLogical(int visualIndex);

    public int logicalToVisual(int logicalIndex);

    public byte getCharLevel(int logicalIndex);

    public boolean isCharLTR(int logicalIndex);

    public int getCharType(int logicalIndex);

    public boolean isCharSpace(int logicalIndex);

    public boolean isCharWhitespace(int logicalIndex);

    public float getCharAngle(int logicalIndex);

    public CoreMetrics getCoreMetricsAt(int logicalIndex);

    public float getCharAscent(int logicalIndex);

    public float getCharDescent(int logicalIndex);

    public float getCharShift(int logicalIndex);

    public float getCharAdvance(int logicalIndex);

    public float getCharXPosition(int logicalIndex);

    public float getCharYPosition(int logicalIndex);

    public float getCharLinePosition(int logicalIndex);

    public float getCharLinePosition(int logicalIndex, boolean leading);

    public boolean caretAtOffsetIsValid(int offset);

    public Rectangle2D getCharBounds(int logicalIndex);

    public void draw(Graphics2D g2, float x, float y);

    public Rectangle2D getVisualBounds();

    public Rectangle2D getItalicBounds();

    public Shape getOutline(AffineTransform tx);

    public String toString();

    public static TextLine fastCreateTextLine(FontRenderContext frc, char[] chars, Font font, CoreMetrics lm, Map<? extends Attribute, ?> attributes);

    public static TextLineComponent[] createComponentsOnRun(int runStart, int runLimit, char[] chars, int[] charsLtoV, byte[] levels, TextLabelFactory factory, Font font, CoreMetrics cm, FontRenderContext frc, Decoration decorator, TextLineComponent[] components, int numComponents);

    public static TextLineComponent[] getComponents(StyledParagraph styledParagraph, char[] chars, int textStart, int textLimit, int[] charsLtoV, byte[] levels, TextLabelFactory factory);

    public static TextLine createLineFromText(char[] chars, StyledParagraph styledParagraph, TextLabelFactory factory, boolean isDirectionLTR, float[] baselineOffsets);

    public static TextLine standardCreateTextLine(FontRenderContext frc, AttributedCharacterIterator text, char[] chars, float[] baselineOffsets);

    static boolean advanceToFirstFont(AttributedCharacterIterator aci);

    static float[] getNormalizedOffsets(float[] baselineOffsets, byte baseline);

    static Font getFontAtCurrentPos(AttributedCharacterIterator aci);

    public TextLine getJustifiedLine(float justificationWidth, float justifyRatio, int justStart, int justLimit);

    public static float getAdvanceBetween(TextLineComponent[] components, int start, int limit);

    LayoutPathImpl getLayoutPath();
}
