/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.BufferedImage;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Locale;
    @Positive
import sun.awt.PlatformGraphicsInfo;
    @Positive
import sun.font.FontManager;
    @Positive
import sun.font.FontManagerFactory;
    @Positive
import sun.java2d.HeadlessGraphicsEnvironment;
    @Positive
import sun.java2d.SunGraphicsEnvironment;
    @Positive
import sun.security.action.GetPropertyAction;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class GraphicsEnvironment {

    @Positive
    protected GraphicsEnvironment() {
    @Positive
    }

    @Positive
    private static final class LocalGE {
    @Positive
    }

    @Positive
    public static GraphicsEnvironment getLocalGraphicsEnvironment();

    @Positive
    public static boolean isHeadless();

    @Positive
    static String getHeadlessMessage();

    @Positive
    static void checkHeadless() throws HeadlessException;

    @Positive
    public boolean isHeadlessInstance();

    @Positive
    public abstract GraphicsDevice[] getScreenDevices() throws HeadlessException;

    @Positive
    public abstract GraphicsDevice getDefaultScreenDevice() throws HeadlessException;

    @Positive
    public abstract Graphics2D createGraphics(BufferedImage img);

    @Positive
    public abstract Font[] getAllFonts();

    @Positive
    public abstract String[] getAvailableFontFamilyNames();

    @Positive
    public abstract String[] getAvailableFontFamilyNames(Locale l);

    @Positive
    public boolean registerFont(Font font);

    @Positive
    public void preferLocaleFonts();

    @Positive
    public void preferProportionalFonts();

    @Positive
    public Point getCenterPoint() throws HeadlessException;

    @Positive
    public Rectangle getMaximumWindowBounds() throws HeadlessException;
    @Positive
}

// CFWR semantic augmentation - variant 0
