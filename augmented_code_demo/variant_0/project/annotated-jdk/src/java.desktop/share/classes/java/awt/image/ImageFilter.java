/*
    @Positive
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class ImageFilter implements ImageConsumer, Cloneable {

    @Positive
    public ImageFilter() {
    @Positive
    }

    @Positive
    protected ImageConsumer consumer;

    @Positive
    public ImageFilter getFilterInstance(ImageConsumer ic);

    @Positive
    public void setDimensions(int width, int height);

    @Positive
    public void setProperties(Hashtable<?, ?> props);

    @Positive
    public void setColorModel(ColorModel model);

    @Positive
    public void setHints(int hints);

    @Positive
    public void setPixels(int x, int y, int w, int h, ColorModel model, byte[] pixels, int off, int scansize);

    @Positive
    public void setPixels(int x, int y, int w, int h, ColorModel model, int[] pixels, int off, int scansize);

    @Positive
    public void imageComplete(int status);

    @Positive
    public void resendTopDownLeftRight(ImageProducer ip);

    @Positive
    public Object clone();
    @Positive
}

// CFWR semantic augmentation - variant 0
