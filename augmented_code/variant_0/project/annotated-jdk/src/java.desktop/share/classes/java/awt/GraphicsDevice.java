/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.ColorModel;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.SunToolkit;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class GraphicsDevice {

    @Positive
    protected GraphicsDevice() {
    @Positive
    }

    @Positive
    public static final int TYPE_RASTER_SCREEN;

    @Positive
    public static final int TYPE_PRINTER;

    @Positive
    public static final int TYPE_IMAGE_BUFFER;

    @Positive
    public static enum WindowTranslucency {

    @Positive
        PERPIXEL_TRANSPARENT, TRANSLUCENT, PERPIXEL_TRANSLUCENT
    @Positive
    }

    @Positive
    public abstract int getType();

    @Positive
    public abstract String getIDstring();

    @Positive
    public abstract GraphicsConfiguration[] getConfigurations();

    @Positive
    public abstract GraphicsConfiguration getDefaultConfiguration();

    @Positive
    public GraphicsConfiguration getBestConfiguration(GraphicsConfigTemplate gct);

    @Positive
    public boolean isFullScreenSupported();

    @Positive
    public void setFullScreenWindow(Window w);

    @Positive
    public Window getFullScreenWindow();

    @Positive
    public boolean isDisplayChangeSupported();

    @Positive
    public void setDisplayMode(DisplayMode dm);

    @Positive
    public DisplayMode getDisplayMode();

    @Positive
    public DisplayMode[] getDisplayModes();

    @Positive
    public int getAvailableAcceleratedMemory();

    @Positive
    public boolean isWindowTranslucencySupported(WindowTranslucency translucencyKind);

    @Positive
    static boolean isWindowShapingSupported();

    @Positive
    static boolean isWindowOpacitySupported();

    @Positive
    boolean isWindowPerpixelTranslucencySupported();

    @Positive
    GraphicsConfiguration getTranslucencyCapableGC();
    @Positive
}
