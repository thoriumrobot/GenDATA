/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2013, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.image.BufferedImage;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.awt.image.VolatileImage;
    @Positive
import java.awt.image.WritableRaster;
    @Positive
import sun.awt.image.SunVolatileImage;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class GraphicsConfiguration {

    @Positive
    protected GraphicsConfiguration() {
    @Positive
    }

    @Positive
    public abstract GraphicsDevice getDevice();

    @Positive
    public BufferedImage createCompatibleImage(int width, int height);

    @Positive
    public BufferedImage createCompatibleImage(int width, int height, int transparency);

    @Positive
    public VolatileImage createCompatibleVolatileImage(int width, int height);

    @Positive
    public VolatileImage createCompatibleVolatileImage(int width, int height, int transparency);

    @Positive
    public VolatileImage createCompatibleVolatileImage(int width, int height, ImageCapabilities caps) throws AWTException;

    @Positive
    public VolatileImage createCompatibleVolatileImage(int width, int height, ImageCapabilities caps, int transparency) throws AWTException;

    @Positive
    public abstract ColorModel getColorModel();

    @Positive
    public abstract ColorModel getColorModel(int transparency);

    @Positive
    public abstract AffineTransform getDefaultTransform();

    @Positive
    public abstract AffineTransform getNormalizingTransform();

    @Positive
    public abstract Rectangle getBounds();

    @Positive
    private static class DefaultBufferCapabilities extends BufferCapabilities {

    @Positive
        public DefaultBufferCapabilities(ImageCapabilities imageCaps) {
    @Positive
        }
    @Positive
    }

    @Positive
    public BufferCapabilities getBufferCapabilities();

    @Positive
    public ImageCapabilities getImageCapabilities();

    @Positive
    public boolean isTranslucencyCapable();
    @Positive
}
