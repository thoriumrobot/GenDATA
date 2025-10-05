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
import java.awt.image.ImageConsumer;
import java.awt.image.ImageProducer;
import java.awt.image.ColorModel;
import java.util.Hashtable;
import java.util.Vector;
import java.util.Enumeration;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class MemoryImageSource implements ImageProducer {

    public MemoryImageSource(int w, int h, ColorModel cm, byte[] pix, int off, int scan) {
    }

    public MemoryImageSource(int w, int h, ColorModel cm, byte[] pix, int off, int scan, Hashtable<?, ?> props) {
    }

    public MemoryImageSource(int w, int h, ColorModel cm, int[] pix, int off, int scan) {
    }

    public MemoryImageSource(int w, int h, ColorModel cm, int[] pix, int off, int scan, Hashtable<?, ?> props) {
    }

    public MemoryImageSource(int w, int h, int[] pix, int off, int scan) {
    }

    public MemoryImageSource(int w, int h, int[] pix, int off, int scan, Hashtable<?, ?> props) {
    }

    public synchronized void addConsumer(ImageConsumer ic);

    public synchronized boolean isConsumer(ImageConsumer ic);

    public synchronized void removeConsumer(ImageConsumer ic);

    public void startProduction(ImageConsumer ic);

    public void requestTopDownLeftRightResend(ImageConsumer ic);

    public synchronized void setAnimated(boolean animated);

    public synchronized void setFullBufferUpdates(boolean fullbuffers);

    public void newPixels();

    public synchronized void newPixels(int x, int y, int w, int h);

    public synchronized void newPixels(int x, int y, int w, int h, boolean framenotify);

    public synchronized void newPixels(byte[] newpix, ColorModel newmodel, int offset, int scansize);

    public synchronized void newPixels(int[] newpix, ColorModel newmodel, int offset, int scansize);
}
