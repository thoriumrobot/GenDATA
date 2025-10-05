/*
 * Copyright (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.ColorModel;
import sun.awt.AppContext;
import sun.awt.SunToolkit;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class GraphicsDevice {

    protected GraphicsDevice() {
    }

    public static final int TYPE_RASTER_SCREEN;

    public static final int TYPE_PRINTER;

    public static final int TYPE_IMAGE_BUFFER;

    public static enum WindowTranslucency {

        PERPIXEL_TRANSPARENT, TRANSLUCENT, PERPIXEL_TRANSLUCENT
    }

    public abstract int getType();

    public abstract String getIDstring();

    public abstract GraphicsConfiguration[] getConfigurations();

    public abstract GraphicsConfiguration getDefaultConfiguration();

    public GraphicsConfiguration getBestConfiguration(GraphicsConfigTemplate gct);

    public boolean isFullScreenSupported();

    public void setFullScreenWindow(Window w);

    public Window getFullScreenWindow();

    public boolean isDisplayChangeSupported();

    public void setDisplayMode(DisplayMode dm);

    public DisplayMode getDisplayMode();

    public DisplayMode[] getDisplayModes();

    public int getAvailableAcceleratedMemory();

    public boolean isWindowTranslucencySupported(WindowTranslucency translucencyKind);

    static boolean isWindowShapingSupported();

    static boolean isWindowOpacitySupported();

    boolean isWindowPerpixelTranslucencySupported();

    GraphicsConfiguration getTranslucencyCapableGC();
}
