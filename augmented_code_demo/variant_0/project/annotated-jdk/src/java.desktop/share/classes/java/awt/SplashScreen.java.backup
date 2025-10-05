/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.io.IOException;
import java.awt.image.*;
import java.net.URL;
import java.net.URLConnection;
import java.io.File;
import sun.util.logging.PlatformLogger;
import sun.awt.image.SunWritableRaster;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public final class SplashScreen {

    @SuppressWarnings("removal")
    public static SplashScreen getSplashScreen();

    public void setImageURL(URL imageURL) throws NullPointerException, IOException, IllegalStateException;

    @SuppressWarnings("deprecation")
    public URL getImageURL() throws IllegalStateException;

    public Rectangle getBounds() throws IllegalStateException;

    public Dimension getSize() throws IllegalStateException;

    public Graphics2D createGraphics() throws IllegalStateException;

    public void update() throws IllegalStateException;

    public void close() throws IllegalStateException;

    static void markClosed();

    public boolean isVisible();
}
