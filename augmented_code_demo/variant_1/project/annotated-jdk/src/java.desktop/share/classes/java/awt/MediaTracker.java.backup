/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.ImageObserver;
import java.io.Serial;
import sun.awt.image.MultiResolutionToolkitImage;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class MediaTracker implements java.io.Serializable {

    public MediaTracker(Component comp) {
    }

    public void addImage(Image image, int id);

    public synchronized void addImage(Image image, int id, int w, int h);

    public static final int LOADING;

    public static final int ABORTED;

    public static final int ERRORED;

    public static final int COMPLETE;

    public boolean checkAll();

    public boolean checkAll(boolean load);

    public synchronized boolean isErrorAny();

    public synchronized Object[] getErrorsAny();

    public void waitForAll() throws InterruptedException;

    public synchronized boolean waitForAll(long ms) throws InterruptedException;

    public int statusAll(boolean load);

    public boolean checkID(int id);

    public boolean checkID(int id, boolean load);

    public synchronized boolean isErrorID(int id);

    public synchronized Object[] getErrorsID(int id);

    public void waitForID(int id) throws InterruptedException;

    public synchronized boolean waitForID(int id, long ms) throws InterruptedException;

    public int statusID(int id, boolean load);

    public synchronized void removeImage(Image image);

    public synchronized void removeImage(Image image, int id);

    public synchronized void removeImage(Image image, int id, int width, int height);

    synchronized void setDone();
}

abstract class MediaEntry {

    abstract Object getMedia();

    static MediaEntry insert(MediaEntry head, MediaEntry me);

    int getID();

    abstract void startLoad();

    void cancel();

    synchronized int getStatus(boolean doLoad, boolean doVerify);

    void setStatus(int flag);
}

@SuppressWarnings("serial")
class ImageMediaEntry extends MediaEntry implements ImageObserver, java.io.Serializable {

    boolean matches(Image img, int w, int h);

    Object getMedia();

    synchronized int getStatus(boolean doLoad, boolean doVerify);

    void startLoad();

    int parseflags(int infoflags);

    public boolean imageUpdate(Image img, int infoflags, int x, int y, int w, int h);
}
