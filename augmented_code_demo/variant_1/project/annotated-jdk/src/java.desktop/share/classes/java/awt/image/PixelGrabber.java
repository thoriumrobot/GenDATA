/*
    @Positive
 * Copyright (c) 1995, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.image;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Hashtable;
    @Positive
import java.awt.image.ImageProducer;
    @Positive
import java.awt.image.ImageConsumer;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.awt.Image;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class PixelGrabber implements ImageConsumer {

    @Positive
    public PixelGrabber(Image img, int x, int y, int w, int h, int[] pix, int off, int scansize) {
    @Positive
    }

    @Positive
    public PixelGrabber(ImageProducer ip, int x, int y, int w, int h, int[] pix, int off, int scansize) {
    @Positive
    }

    @Positive
    public PixelGrabber(Image img, int x, int y, int w, int h, boolean forceRGB) {
    @Positive
    }

    @Positive
    public synchronized void startGrabbing();

    @Positive
    public synchronized void abortGrabbing();

    @Positive
    public boolean grabPixels() throws InterruptedException;

    @Positive
    public synchronized boolean grabPixels(long ms) throws InterruptedException;

    @Positive
    public synchronized int getStatus();

    @Positive
    public synchronized int getWidth();

    @Positive
    public synchronized int getHeight();

    @Positive
    public synchronized Object getPixels();

    @Positive
    public synchronized ColorModel getColorModel();

    @Positive
    public void setDimensions(int width, int height);

    @Positive
    public void setHints(int hints);

    @Positive
    public void setProperties(Hashtable<?, ?> props);

    @Positive
    public void setColorModel(ColorModel model);

    @Positive
    public void setPixels(int srcX, int srcY, int srcW, int srcH, ColorModel model, byte[] pixels, int srcOff, int srcScan);

    @Positive
    public void setPixels(int srcX, int srcY, int srcW, int srcH, ColorModel model, int[] pixels, int srcOff, int srcScan);

    @Positive
    public synchronized void imageComplete(int status);

    @Positive
    public synchronized int status();
    @Positive
}

// CFWR semantic augmentation - variant 1
