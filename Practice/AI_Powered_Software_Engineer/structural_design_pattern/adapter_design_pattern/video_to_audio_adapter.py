
class AudioPlayer:
    def play_audio(self, file_name):
        pass

class VideoPlayer:
    def play_video(self, file_name):
        print(f"Playing video...{file_name}")

class VideoToAudioAdapter(AudioPlayer):
    def __init__(self, video_file):
        self.video_player = video_file
        
    def play_audio(self, file_name):
        self.video_player.play_video(file_name)

if __name__ == "__main__":
    def play_quran(player, file):
        player.play_audio(file)

    video_player = VideoPlayer()
    adapter = VideoToAudioAdapter(video_player)

    play_quran(adapter, 'file.mp3')