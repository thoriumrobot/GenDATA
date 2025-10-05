/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.font.FontRenderContext;
import java.awt.font.LineMetrics;
import java.awt.geom.Rectangle2D;
import java.io.Serial;
import java.text.CharacterIterator;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class FontMetrics implements java.io.Serializable {

    protected Font font;

    protected FontMetrics(Font font) {
    }

    public Font getFont();

    public FontRenderContext getFontRenderContext();

    public int getLeading();

    public int getAscent();

    public int getDescent();

    public int getHeight();

    public int getMaxAscent();

    public int getMaxDescent();

    @Deprecated
    public int getMaxDecent();

    public int getMaxAdvance();

    public int charWidth(int codePoint);

    public int charWidth(char ch);

    public int stringWidth(String str);

    public int charsWidth(char[] data, int off, int len);

    @SuppressWarnings("deprecation")
    public int bytesWidth(byte[] data, int off, int len);

    public int[] getWidths();

    public boolean hasUniformLineMetrics();

    public LineMetrics getLineMetrics(String str, Graphics context);

    public LineMetrics getLineMetrics(String str, int beginIndex, int limit, Graphics context);

    public LineMetrics getLineMetrics(char[] chars, int beginIndex, int limit, Graphics context);

    public LineMetrics getLineMetrics(CharacterIterator ci, int beginIndex, int limit, Graphics context);

    public Rectangle2D getStringBounds(String str, Graphics context);

    public Rectangle2D getStringBounds(String str, int beginIndex, int limit, Graphics context);

    public Rectangle2D getStringBounds(char[] chars, int beginIndex, int limit, Graphics context);

    public Rectangle2D getStringBounds(CharacterIterator ci, int beginIndex, int limit, Graphics context);

    public Rectangle2D getMaxCharBounds(Graphics context);

    public String toString();
}
