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
import java.io.*;
import java.lang.*;
import java.util.*;
import java.awt.image.ImageObserver;
import java.text.AttributedCharacterIterator;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class Graphics {

    protected Graphics() {
    }

    public abstract Graphics create();

    public Graphics create(int x, int y, int width, int height);

    public abstract void translate(int x, int y);

    public abstract Color getColor();

    public abstract void setColor(Color c);

    public abstract void setPaintMode();

    public abstract void setXORMode(Color c1);

    public abstract Font getFont();

    public abstract void setFont(Font font);

    public FontMetrics getFontMetrics();

    public abstract FontMetrics getFontMetrics(Font f);

    public abstract Rectangle getClipBounds();

    public abstract void clipRect(int x, int y, int width, int height);

    public abstract void setClip(int x, int y, int width, int height);

    public abstract Shape getClip();

    public abstract void setClip(Shape clip);

    public abstract void copyArea(int x, int y, int width, int height, int dx, int dy);

    public abstract void drawLine(int x1, int y1, int x2, int y2);

    public abstract void fillRect(int x, int y, int width, int height);

    public void drawRect(int x, int y, int width, int height);

    public abstract void clearRect(int x, int y, int width, int height);

    public abstract void drawRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight);

    public abstract void fillRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight);

    public void draw3DRect(int x, int y, int width, int height, boolean raised);

    public void fill3DRect(int x, int y, int width, int height, boolean raised);

    public abstract void drawOval(int x, int y, int width, int height);

    public abstract void fillOval(int x, int y, int width, int height);

    public abstract void drawArc(int x, int y, int width, int height, int startAngle, int arcAngle);

    public abstract void fillArc(int x, int y, int width, int height, int startAngle, int arcAngle);

    public abstract void drawPolyline(int[] xPoints, int[] yPoints, int nPoints);

    public abstract void drawPolygon(int[] xPoints, int[] yPoints, int nPoints);

    public void drawPolygon(Polygon p);

    public abstract void fillPolygon(int[] xPoints, int[] yPoints, int nPoints);

    public void fillPolygon(Polygon p);

    public abstract void drawString(String str, int x, int y);

    public abstract void drawString(AttributedCharacterIterator iterator, int x, int y);

    public void drawChars(char[] data, int offset, int length, int x, int y);

    @SuppressWarnings("deprecation")
    public void drawBytes(byte[] data, int offset, int length, int x, int y);

    public abstract boolean drawImage(Image img, int x, int y, ImageObserver observer);

    public abstract boolean drawImage(Image img, int x, int y, int width, int height, ImageObserver observer);

    public abstract boolean drawImage(Image img, int x, int y, Color bgcolor, ImageObserver observer);

    public abstract boolean drawImage(Image img, int x, int y, int width, int height, Color bgcolor, ImageObserver observer);

    public abstract boolean drawImage(Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2, ImageObserver observer);

    public abstract boolean drawImage(Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2, Color bgcolor, ImageObserver observer);

    public abstract void dispose();

    @Deprecated()
    public void finalize();

    public String toString();

    @Deprecated
    public Rectangle getClipRect();

    public boolean hitClip(int x, int y, int width, int height);

    public Rectangle getClipBounds(Rectangle r);
}
