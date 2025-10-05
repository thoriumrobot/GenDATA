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
import java.awt.image.ImageConsumer;
    @Positive
import java.awt.image.ImageProducer;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Vector;
    @Positive
import java.util.Enumeration;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class MemoryImageSource implements ImageProducer {

    @Positive
    public MemoryImageSource(int w, int h, ColorModel cm, byte[] pix, int off, int scan) {
    @Positive
    }

    @Positive
    public MemoryImageSource(int w, int h, ColorModel cm, byte[] pix, int off, int scan, Hashtable<?, ?> props) {
    @Positive
    }

    @Positive
    public MemoryImageSource(int w, int h, ColorModel cm, int[] pix, int off, int scan) {
    @Positive
    }

    @Positive
    public MemoryImageSource(int w, int h, ColorModel cm, int[] pix, int off, int scan, Hashtable<?, ?> props) {
    @Positive
    }

    @Positive
    public MemoryImageSource(int w, int h, int[] pix, int off, int scan) {
    @Positive
    }

    @Positive
    public MemoryImageSource(int w, int h, int[] pix, int off, int scan, Hashtable<?, ?> props) {
    @Positive
    }

    @Positive
    public synchronized void addConsumer(ImageConsumer ic);

    @Positive
    public synchronized boolean isConsumer(ImageConsumer ic);

    @Positive
    public synchronized void removeConsumer(ImageConsumer ic);

    @Positive
    public void startProduction(ImageConsumer ic);

    @Positive
    public void requestTopDownLeftRightResend(ImageConsumer ic);

    @Positive
    public synchronized void setAnimated(boolean animated);

    @Positive
    public synchronized void setFullBufferUpdates(boolean fullbuffers);

    @Positive
    public void newPixels();

    @Positive
    public synchronized void newPixels(int x, int y, int w, int h);

    @Positive
    public synchronized void newPixels(int x, int y, int w, int h, boolean framenotify);

    @Positive
    public synchronized void newPixels(byte[] newpix, ColorModel newmodel, int offset, int scansize);

    @Positive
    public synchronized void newPixels(int[] newpix, ColorModel newmodel, int offset, int scansize);
    @Positive
}

// CFWR semantic augmentation - variant 0
