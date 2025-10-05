/*
 * Copyright (c) 1995, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.image;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.Hashtable;
import java.awt.image.ImageProducer;
import java.awt.image.ImageConsumer;
import java.awt.image.ColorModel;
import java.awt.Image;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class PixelGrabber implements ImageConsumer {

    public PixelGrabber(Image img, int x, int y, int w, int h, int[] pix, int off, int scansize) {
    }

    public PixelGrabber(ImageProducer ip, int x, int y, int w, int h, int[] pix, int off, int scansize) {
    }

    public PixelGrabber(Image img, int x, int y, int w, int h, boolean forceRGB) {
    }

    public synchronized void startGrabbing();

    public synchronized void abortGrabbing();

    public boolean grabPixels() throws InterruptedException;

    public synchronized boolean grabPixels(long ms) throws InterruptedException;

    public synchronized int getStatus();

    public synchronized int getWidth();

    public synchronized int getHeight();

    public synchronized Object getPixels();

    public synchronized ColorModel getColorModel();

    public void setDimensions(int width, int height);

    public void setHints(int hints);

    public void setProperties(Hashtable<?, ?> props);

    public void setColorModel(ColorModel model);

    public void setPixels(int srcX, int srcY, int srcW, int srcH, ColorModel model, byte[] pixels, int srcOff, int srcScan);

    public void setPixels(int srcX, int srcY, int srcW, int srcH, ColorModel model, int[] pixels, int srcOff, int srcScan);

    public synchronized void imageComplete(int status);

    public synchronized int status();
}
