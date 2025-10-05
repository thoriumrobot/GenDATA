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
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.guieffect.qual.UIType;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.applet.Applet;
    @Positive
import java.awt.dnd.DropTarget;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.AdjustmentEvent;
    @Positive
import java.awt.event.ComponentEvent;
    @Positive
import java.awt.event.ComponentListener;
    @Positive
import java.awt.event.FocusEvent;
    @Positive
import java.awt.event.FocusListener;
    @Positive
import java.awt.event.HierarchyBoundsListener;
    @Positive
import java.awt.event.HierarchyEvent;
    @Positive
import java.awt.event.HierarchyListener;
    @Positive
import java.awt.event.InputEvent;
    @Positive
import java.awt.event.InputMethodEvent;
    @Positive
import java.awt.event.InputMethodListener;
    @Positive
import java.awt.event.ItemEvent;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.awt.event.KeyListener;
    @Positive
import java.awt.event.MouseEvent;
    @Positive
import java.awt.event.MouseListener;
    @Positive
import java.awt.event.MouseMotionListener;
    @Positive
import java.awt.event.MouseWheelEvent;
    @Positive
import java.awt.event.MouseWheelListener;
    @Positive
import java.awt.event.PaintEvent;
    @Positive
import java.awt.event.TextEvent;
    @Positive
import java.awt.im.InputContext;
    @Positive
import java.awt.im.InputMethodRequests;
    @Positive
import java.awt.image.BufferStrategy;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.awt.image.ImageObserver;
    @Positive
import java.awt.image.ImageProducer;
    @Positive
import java.awt.image.VolatileImage;
    @Positive
import java.awt.peer.ComponentPeer;
    @Positive
import java.awt.peer.ContainerPeer;
    @Positive
import java.awt.peer.LightweightPeer;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.beans.PropertyChangeSupport;
    @Positive
import java.beans.Transient;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.PrintWriter;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EventListener;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.Vector;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleComponent;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.accessibility.AccessibleSelection;
    @Positive
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;
    @Positive
import javax.swing.JComponent;
    @Positive
import javax.swing.JRootPane;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.ComponentFactory;
    @Positive
import sun.awt.ConstrainableGraphics;
    @Positive
import sun.awt.EmbeddedFrame;
    @Positive
import sun.awt.RequestFocusController;
    @Positive
import sun.awt.SubRegionShowable;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.awt.dnd.SunDropTargetEvent;
    @Positive
import sun.awt.im.CompositionArea;
    @Positive
import sun.awt.image.VSyncedBSManager;
    @Positive
import sun.font.FontManager;
    @Positive
import sun.font.FontManagerFactory;
    @Positive
import sun.font.SunFontManager;
    @Positive
import sun.java2d.SunGraphics2D;
    @Positive
import sun.java2d.SunGraphicsEnvironment;
    @Positive
import sun.java2d.pipe.Region;
    @Positive
import sun.java2d.pipe.hw.ExtendedBufferCapabilities;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.swing.SwingAccessor;
    @Positive
import sun.util.logging.PlatformLogger;
    @Positive
import static sun.java2d.pipe.hw.ExtendedBufferCapabilities.VSyncType.VSYNC_DEFAULT;
    @Positive
import static sun.java2d.pipe.hw.ExtendedBufferCapabilities.VSyncType.VSYNC_ON;

    @Positive
@AnnotatedFor({ "guieffect", "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
@UIType
    @Positive
public abstract class Component implements ImageObserver, MenuContainer, Serializable {

    @Positive
    static class AWTTreeLock {
    @Positive
    }

    @Positive
    public static final float TOP_ALIGNMENT;

    @Positive
    public static final float CENTER_ALIGNMENT;

    @Positive
    public static final float BOTTOM_ALIGNMENT;

    @Positive
    public static final float LEFT_ALIGNMENT;

    @Positive
    public static final float RIGHT_ALIGNMENT;

    @Positive
    Object getObjectLock();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    final AccessControlContext getAccessControlContext();

    @Positive
    public enum BaselineResizeBehavior {

    @Positive
        CONSTANT_ASCENT, CONSTANT_DESCENT, CENTER_OFFSET, OTHER
    @Positive
    }

    @Positive
    int getBoundsOp();

    @Positive
    void setBoundsOp(int op);

    @Positive
    protected Component() {
    @Positive
    }

    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    void initializeFocusTraversalKeys();

    @Positive
    @Nullable
    @Positive
    String constructComponentName();

    @Positive
    public String getName();

    @Positive
    public void setName(String name);

    @Positive
    @Nullable
    @Positive
    public Container getParent();

    @Positive
    @Nullable
    @Positive
    final Container getParent_NoClientCode();

    @Positive
    @Nullable
    @Positive
    Container getContainer();

    @Positive
    public synchronized void setDropTarget(DropTarget dt);

    @Positive
    @Nullable
    @Positive
    public synchronized DropTarget getDropTarget();

    @Positive
    @Nullable
    @Positive
    public GraphicsConfiguration getGraphicsConfiguration();

    @Positive
    @Nullable
    @Positive
    final GraphicsConfiguration getGraphicsConfiguration_NoClientCode();

    @Positive
    void setGraphicsConfiguration(GraphicsConfiguration gc);

    @Positive
    final boolean updateGraphicsData(GraphicsConfiguration gc);

    @Positive
    boolean updateChildGraphicsData(GraphicsConfiguration gc);

    @Positive
    void checkGD(String stringID);

    @Positive
    public final Object getTreeLock();

    @Positive
    final void checkTreeLock();

    @Positive
    public Toolkit getToolkit();

    @Positive
    final Toolkit getToolkitImpl();

    @Positive
    final ComponentFactory getComponentFactory();

    @Positive
    public boolean isValid();

    @Positive
    public boolean isDisplayable();

    @Positive
    @Transient
    @Positive
    public boolean isVisible();

    @Positive
    final boolean isVisible_NoClientCode();

    @Positive
    boolean isRecursivelyVisible();

    @Positive
    Point pointRelativeToComponent(Point absolute);

    @Positive
    @Nullable
    @Positive
    Component findUnderMouseInWindow(PointerInfo pi);

    @Positive
    public Point getMousePosition() throws HeadlessException;

    @Positive
    boolean isSameOrAncestorOf(Component comp, boolean allowChildren);

    @Positive
    public boolean isShowing();

    @Positive
    public boolean isEnabled();

    @Positive
    final boolean isEnabledImpl();

    @Positive
    public void setEnabled(boolean b);

    @Positive
    @Deprecated
    @Positive
    public void enable();

    @Positive
    @Deprecated
    @Positive
    public void enable(boolean b);

    @Positive
    @Deprecated
    @Positive
    public void disable();

    @Positive
    public boolean isDoubleBuffered();

    @Positive
    public void enableInputMethods(boolean enable);

    @Positive
    public void setVisible(boolean b);

    @Positive
    @Deprecated
    @Positive
    public void show();

    @Positive
    @Deprecated
    @Positive
    public void show(boolean b);

    @Positive
    boolean containsFocus();

    @Positive
    void clearMostRecentFocusOwnerOnHide();

    @Positive
    void clearCurrentFocusCycleRootOnHide();

    @Positive
    @Deprecated
    @Positive
    public void hide();

    @Positive
    @Transient
    @Positive
    @Nullable
    @Positive
    public Color getForeground();

    @Positive
    public void setForeground(@Nullable Color c);

    @Positive
    public boolean isForegroundSet();

    @Positive
    @Transient
    @Positive
    @Nullable
    @Positive
    public Color getBackground();

    @Positive
    public void setBackground(@Nullable Color c);

    @Positive
    public boolean isBackgroundSet();

    @Positive
    @Transient
    @Positive
    @Nullable
    @Positive
    public Font getFont();

    @Positive
    @Nullable
    @Positive
    final Font getFont_NoClientCode();

    @Positive
    public void setFont(Font f);

    @Positive
    public boolean isFontSet();

    @Positive
    public Locale getLocale();

    @Positive
    public void setLocale(Locale l);

    @Positive
    public ColorModel getColorModel();

    @Positive
    public Point getLocation();

    @Positive
    public Point getLocationOnScreen();

    @Positive
    final Point getLocationOnScreen_NoTreeLock();

    @Positive
    @Deprecated
    @Positive
    public Point location();

    @Positive
    public void setLocation(int x, int y);

    @Positive
    @Deprecated
    @Positive
    public void move(int x, int y);

    @Positive
    public void setLocation(Point p);

    @Positive
    public Dimension getSize();

    @Positive
    @Deprecated
    @Positive
    public Dimension size();

    @Positive
    public void setSize(int width, int height);

    @Positive
    @Deprecated
    @Positive
    public void resize(int width, int height);

    @Positive
    public void setSize(Dimension d);

    @Positive
    @Deprecated
    @Positive
    public void resize(Dimension d);

    @Positive
    public Rectangle getBounds();

    @Positive
    @Deprecated
    @Positive
    public Rectangle bounds();

    @Positive
    public void setBounds(int x, int y, int width, int height);

    @Positive
    @Deprecated
    @Positive
    public void reshape(int x, int y, int width, int height);

    @Positive
    public void setBounds(Rectangle r);

    @Positive
    public int getX();

    @Positive
    public int getY();

    @Positive
    public int getWidth();

    @Positive
    public int getHeight();

    @Positive
    public Rectangle getBounds(Rectangle rv);

    @Positive
    public Dimension getSize(Dimension rv);

    @Positive
    public Point getLocation(Point rv);

    @Positive
    public boolean isOpaque();

    @Positive
    public boolean isLightweight();

    @Positive
    public void setPreferredSize(@Nullable Dimension preferredSize);

    @Positive
    public boolean isPreferredSizeSet();

    @Positive
    public Dimension getPreferredSize();

    @Positive
    @Deprecated
    @Positive
    public Dimension preferredSize();

    @Positive
    public void setMinimumSize(@Nullable Dimension minimumSize);

    @Positive
    public boolean isMinimumSizeSet();

    @Positive
    public Dimension getMinimumSize();

    @Positive
    @Deprecated
    @Positive
    public Dimension minimumSize();

    @Positive
    public void setMaximumSize(@Nullable Dimension maximumSize);

    @Positive
    public boolean isMaximumSizeSet();

    @Positive
    public Dimension getMaximumSize();

    @Positive
    public float getAlignmentX();

    @Positive
    public float getAlignmentY();

    @Positive
    public int getBaseline(int width, int height);

    @Positive
    public BaselineResizeBehavior getBaselineResizeBehavior();

    @Positive
    public void doLayout();

    @Positive
    @Deprecated
    @Positive
    public void layout();

    @Positive
    public void validate();

    @Positive
    public void invalidate();

    @Positive
    void invalidateParent();

    @Positive
    final void invalidateIfValid();

    @Positive
    public void revalidate();

    @Positive
    final void revalidateSynchronously();

    @Positive
    @Nullable
    @Positive
    public Graphics getGraphics();

    @Positive
    @Nullable
    @Positive
    final Graphics getGraphics_NoClientCode();

    @Positive
    public FontMetrics getFontMetrics(Font font);

    @Positive
    public void setCursor(@Nullable Cursor cursor);

    @Positive
    final void updateCursorImmediately();

    @Positive
    public Cursor getCursor();

    @Positive
    final Cursor getCursor_NoClientCode();

    @Positive
    public boolean isCursorSet();

    @Positive
    public void paint(Graphics g);

    @Positive
    public void update(Graphics g);

    @Positive
    public void paintAll(Graphics g);

    @Positive
    void lightweightPaint(Graphics g);

    @Positive
    void paintHeavyweightComponents(Graphics g);

    @Positive
    @SafeEffect
    @Positive
    public void repaint();

    @Positive
    @SafeEffect
    @Positive
    public void repaint(long tm);

    @Positive
    @SafeEffect
    @Positive
    public void repaint(int x, int y, int width, int height);

    @Positive
    @SafeEffect
    @Positive
    public void repaint(long tm, int x, int y, int width, int height);

    @Positive
    public void print(Graphics g);

    @Positive
    public void printAll(Graphics g);

    @Positive
    void lightweightPrint(Graphics g);

    @Positive
    void printHeavyweightComponents(Graphics g);

    @Positive
    public boolean imageUpdate(Image img, int infoflags, int x, int y, int w, int h);

    @Positive
    public Image createImage(ImageProducer producer);

    @Positive
    @Nullable
    @Positive
    public Image createImage(int width, int height);

    @Positive
    @Nullable
    @Positive
    public VolatileImage createVolatileImage(int width, int height);

    @Positive
    @Nullable
    @Positive
    public VolatileImage createVolatileImage(int width, int height, ImageCapabilities caps) throws AWTException;

    @Positive
    public boolean prepareImage(Image image, ImageObserver observer);

    @Positive
    public boolean prepareImage(Image image, int width, int height, ImageObserver observer);

    @Positive
    public int checkImage(Image image, ImageObserver observer);

    @Positive
    public int checkImage(Image image, int width, int height, ImageObserver observer);

    @Positive
    void createBufferStrategy(int numBuffers);

    @Positive
    void createBufferStrategy(int numBuffers, BufferCapabilities caps) throws AWTException;

    @Positive
    private class ProxyCapabilities extends ExtendedBufferCapabilities {
    @Positive
    }

    @Positive
    BufferStrategy getBufferStrategy();

    @Positive
    Image getBackBuffer();

    @Positive
    protected class FlipBufferStrategy extends BufferStrategy {

    @Positive
        protected int numBuffers;

    @Positive
        protected BufferCapabilities caps;

    @Positive
        protected Image drawBuffer;

    @Positive
        protected VolatileImage drawVBuffer;

    @Positive
        protected boolean validatedContents;

    @Positive
        @SuppressWarnings("removal")
    @Positive
        protected FlipBufferStrategy(int numBuffers, BufferCapabilities caps) throws AWTException {
    @Positive
        }

    @Positive
        protected void createBuffers(int numBuffers, BufferCapabilities caps) throws AWTException;

    @Positive
        protected Image getBackBuffer();

    @Positive
        protected void flip(BufferCapabilities.FlipContents flipAction);

    @Positive
        void flipSubRegion(int x1, int y1, int x2, int y2, BufferCapabilities.FlipContents flipAction);

    @Positive
        protected void destroyBuffers();

    @Positive
        public BufferCapabilities getCapabilities();

    @Positive
        public Graphics getDrawGraphics();

    @Positive
        protected void revalidate();

    @Positive
        public boolean contentsLost();

    @Positive
        public boolean contentsRestored();

    @Positive
        public void show();

    @Positive
        void showSubRegion(int x1, int y1, int x2, int y2);

    @Positive
        public void dispose();
    @Positive
    }

    @Positive
    protected class BltBufferStrategy extends BufferStrategy {

    @Positive
        protected BufferCapabilities caps;

    @Positive
        protected VolatileImage[] backBuffers;

    @Positive
        protected boolean validatedContents;

    @Positive
        protected int width;

    @Positive
        protected int height;

    @Positive
        protected BltBufferStrategy(int numBuffers, BufferCapabilities caps) {
    @Positive
        }

    @Positive
        public void dispose();

    @Positive
        protected void createBackBuffers(int numBuffers);

    @Positive
        public BufferCapabilities getCapabilities();

    @Positive
        public Graphics getDrawGraphics();

    @Positive
        Image getBackBuffer();

    @Positive
        public void show();

    @Positive
        void showSubRegion(int x1, int y1, int x2, int y2);

    @Positive
        protected void revalidate();

    @Positive
        void revalidate(boolean checkSize);

    @Positive
        public boolean contentsLost();

    @Positive
        public boolean contentsRestored();
    @Positive
    }

    @Positive
    private class FlipSubRegionBufferStrategy extends FlipBufferStrategy implements SubRegionShowable {

    @Positive
        protected FlipSubRegionBufferStrategy(int numBuffers, BufferCapabilities caps) throws AWTException {
    @Positive
        }

    @Positive
        public void show(int x1, int y1, int x2, int y2);

    @Positive
        public boolean showIfNotLost(int x1, int y1, int x2, int y2);
    @Positive
    }

    @Positive
    private class BltSubRegionBufferStrategy extends BltBufferStrategy implements SubRegionShowable {

    @Positive
        protected BltSubRegionBufferStrategy(int numBuffers, BufferCapabilities caps) {
    @Positive
        }

    @Positive
        public void show(int x1, int y1, int x2, int y2);

    @Positive
        public boolean showIfNotLost(int x1, int y1, int x2, int y2);
    @Positive
    }

    @Positive
    private class SingleBufferStrategy extends BufferStrategy {

    @Positive
        public SingleBufferStrategy(BufferCapabilities caps) {
    @Positive
        }

    @Positive
        public BufferCapabilities getCapabilities();

    @Positive
        public Graphics getDrawGraphics();

    @Positive
        public boolean contentsLost();

    @Positive
        public boolean contentsRestored();

    @Positive
        public void show();
    @Positive
    }

    @Positive
    public void setIgnoreRepaint(boolean ignoreRepaint);

    @Positive
    public boolean getIgnoreRepaint();

    @Positive
    public boolean contains(int x, int y);

    @Positive
    @Deprecated
    @Positive
    public boolean inside(int x, int y);

    @Positive
    public boolean contains(Point p);

    @Positive
    @Nullable
    @Positive
    public Component getComponentAt(int x, int y);

    @Positive
    @Deprecated
    @Positive
    @Nullable
    @Positive
    public Component locate(int x, int y);

    @Positive
    @Nullable
    @Positive
    public Component getComponentAt(Point p);

    @Positive
    @Deprecated
    @Positive
    public void deliverEvent(Event e);

    @Positive
    public final void dispatchEvent(AWTEvent e);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    void dispatchEventImpl(AWTEvent e);

    @Positive
    void autoProcessMouseWheel(MouseWheelEvent e);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    boolean dispatchMouseWheelToAncestor(MouseWheelEvent e);

    @Positive
    boolean areInputMethodsEnabled();

    @Positive
    boolean eventEnabled(AWTEvent e);

    @Positive
    boolean eventTypeEnabled(int type);

    @Positive
    @Deprecated
    @Positive
    public boolean postEvent(Event e);

    @Positive
    public synchronized void addComponentListener(ComponentListener l);

    @Positive
    public synchronized void removeComponentListener(ComponentListener l);

    @Positive
    public synchronized ComponentListener[] getComponentListeners();

    @Positive
    public synchronized void addFocusListener(FocusListener l);

    @Positive
    public synchronized void removeFocusListener(FocusListener l);

    @Positive
    public synchronized FocusListener[] getFocusListeners();

    @Positive
    public void addHierarchyListener(HierarchyListener l);

    @Positive
    public void removeHierarchyListener(HierarchyListener l);

    @Positive
    public synchronized HierarchyListener[] getHierarchyListeners();

    @Positive
    public void addHierarchyBoundsListener(HierarchyBoundsListener l);

    @Positive
    public void removeHierarchyBoundsListener(HierarchyBoundsListener l);

    @Positive
    int numListening(long mask);

    @Positive
    int countHierarchyMembers();

    @Positive
    int createHierarchyEvents(int id, Component changed, Container changedParent, long changeFlags, boolean enabledOnToolkit);

    @Positive
    public synchronized HierarchyBoundsListener[] getHierarchyBoundsListeners();

    @Positive
    void adjustListeningChildrenOnParent(long mask, int num);

    @Positive
    public synchronized void addKeyListener(KeyListener l);

    @Positive
    public synchronized void removeKeyListener(KeyListener l);

    @Positive
    public synchronized KeyListener[] getKeyListeners();

    @Positive
    public synchronized void addMouseListener(MouseListener l);

    @Positive
    public synchronized void removeMouseListener(MouseListener l);

    @Positive
    public synchronized MouseListener[] getMouseListeners();

    @Positive
    public synchronized void addMouseMotionListener(MouseMotionListener l);

    @Positive
    public synchronized void removeMouseMotionListener(MouseMotionListener l);

    @Positive
    public synchronized MouseMotionListener[] getMouseMotionListeners();

    @Positive
    public synchronized void addMouseWheelListener(MouseWheelListener l);

    @Positive
    public synchronized void removeMouseWheelListener(MouseWheelListener l);

    @Positive
    public synchronized MouseWheelListener[] getMouseWheelListeners();

    @Positive
    public synchronized void addInputMethodListener(InputMethodListener l);

    @Positive
    public synchronized void removeInputMethodListener(InputMethodListener l);

    @Positive
    public synchronized InputMethodListener[] getInputMethodListeners();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    public InputMethodRequests getInputMethodRequests();

    @Positive
    @Nullable
    @Positive
    public InputContext getInputContext();

    @Positive
    protected final void enableEvents(long eventsToEnable);

    @Positive
    protected final void disableEvents(long eventsToDisable);

    @Positive
    final boolean isCoalescingEnabled();

    @Positive
    protected AWTEvent coalesceEvents(AWTEvent existingEvent, AWTEvent newEvent);

    @Positive
    protected void processEvent(AWTEvent e);

    @Positive
    protected void processComponentEvent(ComponentEvent e);

    @Positive
    protected void processFocusEvent(FocusEvent e);

    @Positive
    protected void processKeyEvent(KeyEvent e);

    @Positive
    protected void processMouseEvent(MouseEvent e);

    @Positive
    protected void processMouseMotionEvent(MouseEvent e);

    @Positive
    protected void processMouseWheelEvent(MouseWheelEvent e);

    @Positive
    boolean postsOldMouseEvents();

    @Positive
    protected void processInputMethodEvent(InputMethodEvent e);

    @Positive
    protected void processHierarchyEvent(HierarchyEvent e);

    @Positive
    protected void processHierarchyBoundsEvent(HierarchyEvent e);

    @Positive
    @Deprecated
    @Positive
    public boolean handleEvent(Event evt);

    @Positive
    @Deprecated
    @Positive
    public boolean mouseDown(Event evt, int x, int y);

    @Positive
    @Deprecated
    @Positive
    public boolean mouseDrag(Event evt, int x, int y);

    @Positive
    @Deprecated
    @Positive
    public boolean mouseUp(Event evt, int x, int y);

    @Positive
    @Deprecated
    @Positive
    public boolean mouseMove(Event evt, int x, int y);

    @Positive
    @Deprecated
    @Positive
    public boolean mouseEnter(Event evt, int x, int y);

    @Positive
    @Deprecated
    @Positive
    public boolean mouseExit(Event evt, int x, int y);

    @Positive
    @Deprecated
    @Positive
    public boolean keyDown(Event evt, int key);

    @Positive
    @Deprecated
    @Positive
    public boolean keyUp(Event evt, int key);

    @Positive
    @Deprecated
    @Positive
    public boolean action(Event evt, Object what);

    @Positive
    public void addNotify();

    @Positive
    public void removeNotify();

    @Positive
    @Deprecated
    @Positive
    public boolean gotFocus(Event evt, Object what);

    @Positive
    @Deprecated
    @Positive
    public boolean lostFocus(Event evt, Object what);

    @Positive
    @Deprecated
    @Positive
    public boolean isFocusTraversable();

    @Positive
    public boolean isFocusable();

    @Positive
    public void setFocusable(boolean focusable);

    @Positive
    final boolean isFocusTraversableOverridden();

    @Positive
    public void setFocusTraversalKeys(int id, Set<? extends AWTKeyStroke> keystrokes);

    @Positive
    public Set<AWTKeyStroke> getFocusTraversalKeys(int id);

    @Positive
    final void setFocusTraversalKeys_NoIDCheck(int id, Set<? extends AWTKeyStroke> keystrokes);

    @Positive
    final Set<AWTKeyStroke> getFocusTraversalKeys_NoIDCheck(int id);

    @Positive
    public boolean areFocusTraversalKeysSet(int id);

    @Positive
    public void setFocusTraversalKeysEnabled(boolean focusTraversalKeysEnabled);

    @Positive
    public boolean getFocusTraversalKeysEnabled();

    @Positive
    public void requestFocus();

    @Positive
    public void requestFocus(FocusEvent.Cause cause);

    @Positive
    protected boolean requestFocus(boolean temporary);

    @Positive
    protected boolean requestFocus(boolean temporary, FocusEvent.Cause cause);

    @Positive
    public boolean requestFocusInWindow();

    @Positive
    public boolean requestFocusInWindow(FocusEvent.Cause cause);

    @Positive
    protected boolean requestFocusInWindow(boolean temporary);

    @Positive
    boolean requestFocusInWindow(boolean temporary, FocusEvent.Cause cause);

    @Positive
    final boolean requestFocusHelper(boolean temporary, boolean focusedWindowChangeAllowed);

    @Positive
    final boolean requestFocusHelper(boolean temporary, boolean focusedWindowChangeAllowed, FocusEvent.Cause cause);

    @Positive
    private static class DummyRequestFocusController implements RequestFocusController {

    @Positive
        public boolean acceptRequestFocus(Component from, Component to, boolean temporary, boolean focusedWindowChangeAllowed, FocusEvent.Cause cause);
    @Positive
    }

    @Positive
    static synchronized void setRequestFocusController(RequestFocusController requestController);

    @Positive
    @Nullable
    @Positive
    public Container getFocusCycleRootAncestor();

    @Positive
    public boolean isFocusCycleRoot(Container container);

    @Positive
    Container getTraversalRoot();

    @Positive
    public void transferFocus();

    @Positive
    @Deprecated
    @Positive
    public void nextFocus();

    @Positive
    boolean transferFocus(boolean clearOnFailure);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    final Component getNextFocusCandidate();

    @Positive
    public void transferFocusBackward();

    @Positive
    boolean transferFocusBackward(boolean clearOnFailure);

    @Positive
    public void transferFocusUpCycle();

    @Positive
    public boolean hasFocus();

    @Positive
    public boolean isFocusOwner();

    @Positive
    void setAutoFocusTransferOnDisposal(boolean value);

    @Positive
    boolean isAutoFocusTransferOnDisposal();

    @Positive
    public void add(PopupMenu popup);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public void remove(MenuComponent popup);

    @Positive
    protected String paramString();

    @Positive
    public String toString();

    @Positive
    public void list();

    @Positive
    public void list(PrintStream out);

    @Positive
    public void list(PrintStream out, int indent);

    @Positive
    public void list(PrintWriter out);

    @Positive
    public void list(PrintWriter out, int indent);

    @Positive
    final Container getNativeContainer();

    @Positive
    public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public void removePropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public PropertyChangeListener[] getPropertyChangeListeners();

    @Positive
    public void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

    @Positive
    public void removePropertyChangeListener(String propertyName, PropertyChangeListener listener);

    @Positive
    public PropertyChangeListener[] getPropertyChangeListeners(String propertyName);

    @Positive
    protected void firePropertyChange(String propertyName, @Nullable Object oldValue, @Nullable Object newValue);

    @Positive
    protected void firePropertyChange(String propertyName, boolean oldValue, boolean newValue);

    @Positive
    protected void firePropertyChange(String propertyName, int oldValue, int newValue);

    @Positive
    public void firePropertyChange(String propertyName, byte oldValue, byte newValue);

    @Positive
    public void firePropertyChange(String propertyName, char oldValue, char newValue);

    @Positive
    public void firePropertyChange(String propertyName, short oldValue, short newValue);

    @Positive
    public void firePropertyChange(String propertyName, long oldValue, long newValue);

    @Positive
    public void firePropertyChange(String propertyName, float oldValue, float newValue);

    @Positive
    public void firePropertyChange(String propertyName, double oldValue, double newValue);

    @Positive
    public void setComponentOrientation(ComponentOrientation o);

    @Positive
    public ComponentOrientation getComponentOrientation();

    @Positive
    public void applyComponentOrientation(ComponentOrientation orientation);

    @Positive
    final boolean canBeFocusOwner();

    @Positive
    final boolean canBeFocusOwnerRecursively();

    @Positive
    final void relocateComponent();

    @Positive
    @Nullable
    @Positive
    Window getContainingWindow();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected AccessibleContext accessibleContext;

    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    protected abstract class AccessibleAWTComponent extends AccessibleContext implements Serializable, AccessibleComponent {

    @Positive
        protected AccessibleAWTComponent() {
    @Positive
        }

    @Positive
        @SuppressWarnings("serial")
    @Positive
        protected ComponentListener accessibleAWTComponentHandler;

    @Positive
        @SuppressWarnings("serial")
    @Positive
        protected FocusListener accessibleAWTFocusHandler;

    @Positive
        protected class AccessibleAWTComponentHandler implements ComponentListener, Serializable {

    @Positive
            protected AccessibleAWTComponentHandler() {
    @Positive
            }

    @Positive
            public void componentHidden(ComponentEvent e);

    @Positive
            public void componentShown(ComponentEvent e);

    @Positive
            public void componentMoved(ComponentEvent e);

    @Positive
            public void componentResized(ComponentEvent e);
    @Positive
        }

    @Positive
        protected class AccessibleAWTFocusHandler implements FocusListener, Serializable {

    @Positive
            protected AccessibleAWTFocusHandler() {
    @Positive
            }

    @Positive
            public void focusGained(FocusEvent event);

    @Positive
            public void focusLost(FocusEvent event);
    @Positive
        }

    @Positive
        public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
        public void removePropertyChangeListener(PropertyChangeListener listener);

    @Positive
        public String getAccessibleName();

    @Positive
        public String getAccessibleDescription();

    @Positive
        public AccessibleRole getAccessibleRole();

    @Positive
        public AccessibleStateSet getAccessibleStateSet();

    @Positive
        public Accessible getAccessibleParent();

    @Positive
        public int getAccessibleIndexInParent();

    @Positive
        public int getAccessibleChildrenCount();

    @Positive
        public Accessible getAccessibleChild(int i);

    @Positive
        public Locale getLocale();

    @Positive
        public AccessibleComponent getAccessibleComponent();

    @Positive
        public Color getBackground();

    @Positive
        public void setBackground(Color c);

    @Positive
        public Color getForeground();

    @Positive
        public void setForeground(Color c);

    @Positive
        public Cursor getCursor();

    @Positive
        public void setCursor(Cursor cursor);

    @Positive
        public Font getFont();

    @Positive
        public void setFont(Font f);

    @Positive
        public FontMetrics getFontMetrics(Font f);

    @Positive
        public boolean isEnabled();

    @Positive
        public void setEnabled(boolean b);

    @Positive
        public boolean isVisible();

    @Positive
        public void setVisible(boolean b);

    @Positive
        public boolean isShowing();

    @Positive
        public boolean contains(Point p);

    @Positive
        public Point getLocationOnScreen();

    @Positive
        public Point getLocation();

    @Positive
        public void setLocation(Point p);

    @Positive
        public Rectangle getBounds();

    @Positive
        public void setBounds(Rectangle r);

    @Positive
        public Dimension getSize();

    @Positive
        public void setSize(Dimension d);

    @Positive
        public Accessible getAccessibleAt(Point p);

    @Positive
        public boolean isFocusTraversable();

    @Positive
        public void requestFocus();

    @Positive
        public void addFocusListener(FocusListener l);

    @Positive
        public void removeFocusListener(FocusListener l);
    @Positive
    }

    @Positive
    int getAccessibleIndexInParent();

    @Positive
    AccessibleStateSet getAccessibleStateSet();

    @Positive
    static boolean isInstanceOf(Object obj, String className);

    @Positive
    final boolean areBoundsValid();

    @Positive
    void applyCompoundShape(Region shape);

    @Positive
    Point getLocationOnWindow();

    @Positive
    final Region getNormalShape();

    @Positive
    Region getOpaqueShape();

    @Positive
    final int getSiblingIndexAbove();

    @Positive
    @Nullable
    @Positive
    final ComponentPeer getHWPeerAboveMe();

    @Positive
    final int getSiblingIndexBelow();

    @Positive
    final boolean isNonOpaqueForMixing();

    @Positive
    void applyCurrentShape();

    @Positive
    final void subtractAndApplyShape(Region s);

    @Positive
    final void subtractAndApplyShapeBelowMe();

    @Positive
    void mixOnShowing();

    @Positive
    void mixOnHiding(boolean isLightweight);

    @Positive
    void mixOnReshaping();

    @Positive
    void mixOnZOrderChanging(int oldZorder, int newZorder);

    @Positive
    void mixOnValidating();

    @Positive
    final boolean isMixingNeeded();

    @Positive
    public void setMixingCutoutShape(@Nullable Shape shape);

    @Positive
    void updateZOrder();
    @Positive
}

// CFWR semantic augmentation - variant 0
