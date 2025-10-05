/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.ImageObserver;
    @Positive
import java.io.Serial;
    @Positive
import sun.awt.image.MultiResolutionToolkitImage;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class MediaTracker implements java.io.Serializable {

    @Positive
    public MediaTracker(Component comp) {
    @Positive
    }

    @Positive
    public void addImage(Image image, int id);

    @Positive
    public synchronized void addImage(Image image, int id, int w, int h);

    @Positive
    public static final int LOADING;

    @Positive
    public static final int ABORTED;

    @Positive
    public static final int ERRORED;

    @Positive
    public static final int COMPLETE;

    @Positive
    public boolean checkAll();

    @Positive
    public boolean checkAll(boolean load);

    @Positive
    public synchronized boolean isErrorAny();

    @Positive
    public synchronized Object[] getErrorsAny();

    @Positive
    public void waitForAll() throws InterruptedException;

    @Positive
    public synchronized boolean waitForAll(long ms) throws InterruptedException;

    @Positive
    public int statusAll(boolean load);

    @Positive
    public boolean checkID(int id);

    @Positive
    public boolean checkID(int id, boolean load);

    @Positive
    public synchronized boolean isErrorID(int id);

    @Positive
    public synchronized Object[] getErrorsID(int id);

    @Positive
    public void waitForID(int id) throws InterruptedException;

    @Positive
    public synchronized boolean waitForID(int id, long ms) throws InterruptedException;

    @Positive
    public int statusID(int id, boolean load);

    @Positive
    public synchronized void removeImage(Image image);

    @Positive
    public synchronized void removeImage(Image image, int id);

    @Positive
    public synchronized void removeImage(Image image, int id, int width, int height);

    @Positive
    synchronized void setDone();
    @Positive
}

    @Positive
abstract class MediaEntry {

    @Positive
    abstract Object getMedia();

    @Positive
    static MediaEntry insert(MediaEntry head, MediaEntry me);

    @Positive
    int getID();

    @Positive
    abstract void startLoad();

    @Positive
    void cancel();

    @Positive
    synchronized int getStatus(boolean doLoad, boolean doVerify);

    @Positive
    void setStatus(int flag);
    @Positive
}

    @Positive
@SuppressWarnings("serial")
    @Positive
class ImageMediaEntry extends MediaEntry implements ImageObserver, java.io.Serializable {

    @Positive
    boolean matches(Image img, int w, int h);

    @Positive
    Object getMedia();

    @Positive
    synchronized int getStatus(boolean doLoad, boolean doVerify);

    @Positive
    void startLoad();

    @Positive
    int parseflags(int infoflags);

    @Positive
    public boolean imageUpdate(Image img, int infoflags, int x, int y, int w, int h);
    @Positive
}

// CFWR semantic augmentation - variant 1
