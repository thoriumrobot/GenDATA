/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.lang.*;
    @Positive
import java.util.*;
    @Positive
import java.awt.image.ImageObserver;
    @Positive
import java.text.AttributedCharacterIterator;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Graphics {

    @Positive
    protected Graphics() {
    @Positive
    }

    @Positive
    public abstract Graphics create();

    @Positive
    public Graphics create(int x, int y, int width, int height);

    @Positive
    public abstract void translate(int x, int y);

    @Positive
    public abstract Color getColor();

    @Positive
    public abstract void setColor(Color c);

    @Positive
    public abstract void setPaintMode();

    @Positive
    public abstract void setXORMode(Color c1);

    @Positive
    public abstract Font getFont();

    @Positive
    public abstract void setFont(Font font);

    @Positive
    public FontMetrics getFontMetrics();

    @Positive
    public abstract FontMetrics getFontMetrics(Font f);

    @Positive
    public abstract Rectangle getClipBounds();

    @Positive
    public abstract void clipRect(int x, int y, int width, int height);

    @Positive
    public abstract void setClip(int x, int y, int width, int height);

    @Positive
    public abstract Shape getClip();

    @Positive
    public abstract void setClip(Shape clip);

    @Positive
    public abstract void copyArea(int x, int y, int width, int height, int dx, int dy);

    @Positive
    public abstract void drawLine(int x1, int y1, int x2, int y2);

    @Positive
    public abstract void fillRect(int x, int y, int width, int height);

    @Positive
    public void drawRect(int x, int y, int width, int height);

    @Positive
    public abstract void clearRect(int x, int y, int width, int height);

    @Positive
    public abstract void drawRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight);

    @Positive
    public abstract void fillRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight);

    @Positive
    public void draw3DRect(int x, int y, int width, int height, boolean raised);

    @Positive
    public void fill3DRect(int x, int y, int width, int height, boolean raised);

    @Positive
    public abstract void drawOval(int x, int y, int width, int height);

    @Positive
    public abstract void fillOval(int x, int y, int width, int height);

    @Positive
    public abstract void drawArc(int x, int y, int width, int height, int startAngle, int arcAngle);

    @Positive
    public abstract void fillArc(int x, int y, int width, int height, int startAngle, int arcAngle);

    @Positive
    public abstract void drawPolyline(int[] xPoints, int[] yPoints, int nPoints);

    @Positive
    public abstract void drawPolygon(int[] xPoints, int[] yPoints, int nPoints);

    @Positive
    public void drawPolygon(Polygon p);

    @Positive
    public abstract void fillPolygon(int[] xPoints, int[] yPoints, int nPoints);

    @Positive
    public void fillPolygon(Polygon p);

    @Positive
    public abstract void drawString(String str, int x, int y);

    @Positive
    public abstract void drawString(AttributedCharacterIterator iterator, int x, int y);

    @Positive
    public void drawChars(char[] data, int offset, int length, int x, int y);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public void drawBytes(byte[] data, int offset, int length, int x, int y);

    @Positive
    public abstract boolean drawImage(Image img, int x, int y, ImageObserver observer);

    @Positive
    public abstract boolean drawImage(Image img, int x, int y, int width, int height, ImageObserver observer);

    @Positive
    public abstract boolean drawImage(Image img, int x, int y, Color bgcolor, ImageObserver observer);

    @Positive
    public abstract boolean drawImage(Image img, int x, int y, int width, int height, Color bgcolor, ImageObserver observer);

    @Positive
    public abstract boolean drawImage(Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2, ImageObserver observer);

    @Positive
    public abstract boolean drawImage(Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2, Color bgcolor, ImageObserver observer);

    @Positive
    public abstract void dispose();

    @Positive
    @Deprecated()
    @Positive
    public void finalize();

    @Positive
    public String toString();

    @Positive
    @Deprecated
    @Positive
    public Rectangle getClipRect();

    @Positive
    public boolean hitClip(int x, int y, int width, int height);

    @Positive
    public Rectangle getClipBounds(Rectangle r);
    @Positive
}
