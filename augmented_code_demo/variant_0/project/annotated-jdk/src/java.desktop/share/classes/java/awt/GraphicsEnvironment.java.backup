/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.BufferedImage;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Locale;
import sun.awt.PlatformGraphicsInfo;
import sun.font.FontManager;
import sun.font.FontManagerFactory;
import sun.java2d.HeadlessGraphicsEnvironment;
import sun.java2d.SunGraphicsEnvironment;
import sun.security.action.GetPropertyAction;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class GraphicsEnvironment {

    protected GraphicsEnvironment() {
    }

    private static final class LocalGE {
    }

    public static GraphicsEnvironment getLocalGraphicsEnvironment();

    public static boolean isHeadless();

    static String getHeadlessMessage();

    static void checkHeadless() throws HeadlessException;

    public boolean isHeadlessInstance();

    public abstract GraphicsDevice[] getScreenDevices() throws HeadlessException;

    public abstract GraphicsDevice getDefaultScreenDevice() throws HeadlessException;

    public abstract Graphics2D createGraphics(BufferedImage img);

    public abstract Font[] getAllFonts();

    public abstract String[] getAvailableFontFamilyNames();

    public abstract String[] getAvailableFontFamilyNames(Locale l);

    public boolean registerFont(Font font);

    public void preferLocaleFonts();

    public void preferProportionalFonts();

    public Point getCenterPoint() throws HeadlessException;

    public Rectangle getMaximumWindowBounds() throws HeadlessException;
}
